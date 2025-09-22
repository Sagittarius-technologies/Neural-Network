#!/usr/bin/env python3
"""
main.py - FastAPI server for eDNA NN API
"""
import sys
import os
import uuid
import json
import time
import shutil
import zipfile
import traceback
import io
from pathlib import Path
from typing import Dict, Any
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import csv as _csv

# Ensure pipeline module is importable (your pipeline.py should be in same dir)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import pipeline  # your pipeline module (must expose train_from_labeled_fasta, cluster_and_predict)

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "runs"
MODELS_DIR = RUNS_DIR / "models"
PREDICTIONS_DIR = RUNS_DIR / "predictions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="eDNA NN API", version="1.0.0")

# --- CORS Middleware ---
# Replace "*" with explicit origins (frontend dev + deployed frontend)
ALLOWED_ORIGINS = [
    "http://localhost:5173",              # dev front-end origin
    "http://localhost:3000",              # optional (if you use CRA)
    "https://your-frontend-domain.com",   # replace with your deployed frontend domain if any
    "https://neural-network-3.onrender.com" # optional if you host frontend here (otherwise remove)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def save_uploaded_file(upload_file: UploadFile, destination: Path):
    """Save uploaded file to destination path."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return True
    except Exception as e:
        print(f"Error saving file {destination}: {e}")
        return False
    finally:
        try:
            upload_file.file.close()
        except:
            pass

def write_job_status(run_dir: Path, status_data: Dict[str, Any]):
    """Write job status to JSON file."""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        status_file = run_dir / "status.json"
        with status_file.open("w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=4)
    except Exception as e:
        print(f"Error writing status file: {e}")

def validate_model_files(model_dir: Path) -> bool:
    """Validate that all required model files exist."""
    required_files = [
        "kmer_vectorizer.joblib",
        "reducer.joblib",
        "scaler.joblib",
        "label_encoder.joblib",
        "mlp_model.joblib"
    ]

    for file_name in required_files:
        if not (model_dir / file_name).exists():
            print(f"Missing required model file: {file_name} in {model_dir}")
            return False
    return True

def _to_number_maybe(v):
    """Return float(v) if convertible, otherwise None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return None
    return None

def pick_latest_completed_model() -> str:
    """Return the run_id (directory name) of the most recent completed model, or empty string."""
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        status_file = model_dir / "status.json"
        if not status_file.exists():
            continue
        try:
            with status_file.open("r", encoding="utf-8") as f:
                sd = json.load(f)
            if sd.get("status") == "completed":
                raw_end = sd.get("end_time")
                end_num = _to_number_maybe(raw_end)
                end_sort = end_num if end_num is not None else 0
                models.append((end_sort, sd.get("run_id", model_dir.name)))
        except Exception:
            continue
    if not models:
        return ""
    models.sort(key=lambda x: x[0], reverse=True)
    return models[0][1]

# --- Background Task Logic ---
def _collect_prediction_outputs(prediction_dir: Path, run_id: str):
    expected_outputs = [
        "cluster_medoid_predictions.csv",
        "sequence_cluster_assignments.csv",
        "species_abundance_by_reads.csv",
        "cluster_scatter.png",
        "species_abundance_bar.png",
        "species_composition_pie.png"
    ]

    for fname in expected_outputs:
        src = Path.cwd() / fname
        dest = prediction_dir / fname
        if src.exists() and not dest.exists():
            try:
                shutil.move(str(src), str(dest))
                print(f"[{run_id}] Moved {src} -> {dest}")
            except Exception as e:
                print(f"[{run_id}] Could not move {src} -> {dest}: {e}")

    keywords = ["scatter", "abundance", "composition", "medoid", "cluster"]
    for ext in ("*.png", "*.csv"):
        for p in Path.cwd().glob(ext):
            lower = p.name.lower()
            if any(k in lower for k in keywords):
                dest = prediction_dir / p.name
                if not dest.exists():
                    try:
                        shutil.move(str(p), str(dest))
                        print(f"[{run_id}] Fallback moved {p} -> {dest}")
                    except Exception as e:
                        print(f"[{run_id}] Fallback move failed for {p}: {e}")

    files_here = []
    try:
        files_here = [p.name for p in prediction_dir.iterdir() if p.is_file()]
    except Exception as e:
        print(f"[{run_id}] Could not list prediction dir files: {e}")
    return files_here

def run_training_task(run_id: str, fasta_path_str: str, params: Dict[str, Any]):
    model_dir = MODELS_DIR / run_id
    fasta_path = Path(fasta_path_str)

    status_data = {
        "run_id": run_id,
        "status": "running",
        "start_time": time.time(),
        "end_time": None,
        "error": None,
        "parameters": {**params, "filename": fasta_path.name},
        "job_type": "training"
    }
    write_job_status(model_dir, status_data)

    print(f"[{run_id}] Starting training with parameters: {params}")

    try:
        if not fasta_path.exists():
            raise FileNotFoundError(f"Training file not found: {fasta_path}")
        if fasta_path.stat().st_size == 0:
            raise ValueError("Training file is empty")

        model_path = pipeline.train_from_labeled_fasta(
            train_fasta=str(fasta_path),
            outdir=str(model_dir),
            k=int(params["k"]),
            pca_comp=int(params["pca_comp"]),
            epochs=int(params["epochs"]),
            centroid_percentile=float(params["centroid_percentile"])
        )

        if not validate_model_files(model_dir):
            raise RuntimeError("Training completed but some model files are missing")

        status_data["status"] = "completed"
        status_data["model_path"] = model_path
        print(f"[{run_id}] Training completed successfully.")

    except Exception as e:
        error_str = str(e)
        traceback_str = traceback.format_exc()
        status_data.update({"status": "failed", "error": error_str, "traceback": traceback_str})
        print(f"[{run_id}] Training failed: {error_str}\nTraceback:\n{traceback_str}")

    finally:
        status_data["end_time"] = time.time()
        try:
            status_data["outputs"] = [p.name for p in model_dir.iterdir() if p.is_file()]
        except Exception:
            status_data["outputs"] = []
        write_job_status(model_dir, status_data)

        try:
            if fasta_path.exists() and fasta_path.name.startswith("uploaded"):
                fasta_path.unlink()
        except:
            pass

def run_prediction_task(run_id: str, model_run_id: str, fasta_path_str: str, params: Dict[str, Any]):
    prediction_dir = PREDICTIONS_DIR / run_id
    model_dir = MODELS_DIR / model_run_id if model_run_id else None
    fasta_path = Path(fasta_path_str)

    prediction_dir.mkdir(parents=True, exist_ok=True)

    status_data = {
        "run_id": run_id,
        "status": "running",
        "start_time": time.time(),
        "end_time": None,
        "error": None,
        "parameters": {**params, "model_run_id": model_run_id, "filename": fasta_path.name},
        "job_type": "prediction"
    }
    write_job_status(prediction_dir, status_data)

    print(f"[{run_id}] Starting prediction using model {model_run_id} with parameters: {params}")

    try:
        if not fasta_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {fasta_path}")
        if fasta_path.stat().st_size == 0:
            raise ValueError("Prediction file is empty")
        if not model_dir or not model_dir.exists():
            raise RuntimeError(f"Model directory not found: {model_run_id}")
        if not validate_model_files(model_dir):
            raise RuntimeError(f"Model directory {model_run_id} is missing required files")

        assign_df, medoid_df, labels = pipeline.cluster_and_predict(
            raw_fasta=str(fasta_path),
            model_dir=str(model_dir),
            outdir=str(prediction_dir),
            k=int(params.get("k", 4)),
            pca_comp=int(params.get("pca_comp", 50)),
            cluster_method=str(params.get("cluster_method", "kmeans")),
            dbscan_eps=float(params.get("dbscan_eps", 0.5)),
            dbscan_min=int(params.get("dbscan_min", 5)),
            kmeans_n=int(params.get("kmeans_n", 10)),
            threshold=float(params.get("threshold", 0.7))
        )

        files_here = _collect_prediction_outputs(prediction_dir, run_id)
        status_data["outputs"] = files_here

        expected = [
            "cluster_medoid_predictions.csv",
            "cluster_scatter.png",
            "species_abundance_bar.png",
            "species_composition_pie.png"
        ]
        missing = [e for e in expected if e not in files_here]
        if missing:
            print(f"[{run_id}] Warning: missing expected outputs: {missing}")

        status_data["status"] = "completed"
        status_data["results_info"] = {
            "total_sequences": int(len(assign_df)) if assign_df is not None else 0,
            "num_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)) if labels is not None else 0,
            "medoids_found": int(len(medoid_df)) if medoid_df is not None else 0
        }
        print(f"[{run_id}] Prediction completed successfully. Files: {files_here}")

    except Exception as e:
        error_str = str(e)
        traceback_str = traceback.format_exc()
        status_data.update({"status": "failed", "error": error_str, "traceback": traceback_str})
        print(f"[{run_id}] Prediction failed: {error_str}\nTraceback:\n{traceback_str}")

    finally:
        status_data["end_time"] = time.time()
        try:
            status_data["outputs"] = status_data.get("outputs") or [p.name for p in prediction_dir.iterdir() if p.is_file()]
        except Exception:
            status_data["outputs"] = []
        write_job_status(prediction_dir, status_data)

        try:
            if fasta_path.exists() and fasta_path.name.startswith("uploaded"):
                fasta_path.unlink()
        except:
            pass

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the eDNA NN API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "train": "POST /train",
            "predict": "POST /predict",
            "status": "GET /runs/{run_id}",
            "models": "GET /models",
            "download": "GET /runs/{run_id}/download"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    fasta: UploadFile = File(...),
    k: int = Form(4, description="K-mer size"),
    pca_comp: int = Form(50, description="PCA components"),
    epochs: int = Form(50, description="Training epochs"),
    centroid_percentile: float = Form(95.0, description="Centroid distance percentile")
):
    try:
        if not fasta.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        if not fasta.filename.lower().endswith(('.fasta', '.fa', '.fas', '.txt')):
            raise HTTPException(status_code=400, detail="File must be a FASTA file (.fasta, .fa, .fas) or .txt")
        if k < 1 or k > 10:
            raise HTTPException(status_code=400, detail="K-mer size must be between 1 and 10")
        if pca_comp < 1 or pca_comp > 1000:
            raise HTTPException(status_code=400, detail="PCA components must be between 1 and 1000")
        if epochs < 1 or epochs > 1000:
            raise HTTPException(status_code=400, detail="Epochs must be between 1 and 1000")
        if centroid_percentile < 50 or centroid_percentile > 99.9:
            raise HTTPException(status_code=400, detail="Centroid percentile must be between 50 and 99.9")

        run_id = f"train_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        model_dir = MODELS_DIR / run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = model_dir / fasta.filename
        if not save_uploaded_file(fasta, fasta_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

        params = {"k": k, "pca_comp": pca_comp, "epochs": epochs, "centroid_percentile": centroid_percentile}
        background_tasks.add_task(run_training_task, run_id, str(fasta_path), params)

        return {"message": "Training job started successfully", "run_id": run_id, "parameters": params, "status_endpoint": f"/runs/{run_id}"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Training endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    raw_fasta: UploadFile = File(...),
    model_run_id: str = Form("", description="Model run ID to use for prediction (optional)"),
    cluster_method: str = Form("kmeans", description="Clustering method: kmeans or dbscan"),
    kmeans_n: int = Form(10, description="Number of clusters for K-means"),
    dbscan_eps: float = Form(0.5, description="DBSCAN epsilon parameter"),
    dbscan_min: int = Form(5, description="DBSCAN min samples parameter"),
    threshold: float = Form(0.7, description="Confidence threshold for predictions")
):
    try:
        if not raw_fasta.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        if not raw_fasta.filename.lower().endswith(('.fasta', '.fa', '.fas', '.txt')):
            raise HTTPException(status_code=400, detail="File must be a FASTA file (.fasta, .fa, .fas) or .txt")
        if cluster_method.lower() not in ["kmeans", "dbscan"]:
            raise HTTPException(status_code=400, detail="Cluster method must be 'kmeans' or 'dbscan'")
        if kmeans_n < 1 or kmeans_n > 100:
            raise HTTPException(status_code=400, detail="K-means clusters must be between 1 and 100")
        if dbscan_eps <= 0 or dbscan_eps > 10:
            raise HTTPException(status_code=400, detail="DBSCAN eps must be between 0 and 10")
        if dbscan_min < 1 or dbscan_min > 50:
            raise HTTPException(status_code=400, detail="DBSCAN min samples must be between 1 and 50")
        if threshold < 0 or threshold > 1:
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

        chosen_model = model_run_id.strip()
        if not chosen_model:
            chosen_model = pick_latest_completed_model()
            if not chosen_model:
                raise HTTPException(status_code=404, detail="No completed trained model found on server. Please train a model first.")

        model_dir = MODELS_DIR / chosen_model
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model with run_id '{chosen_model}' not found")

        status_file = model_dir / "status.json"
        if status_file.exists():
            try:
                with status_file.open("r", encoding="utf-8") as f:
                    model_status = json.load(f)
                if model_status.get("status") != "completed":
                    raise HTTPException(status_code=400, detail="Model training is not completed yet")
            except HTTPException:
                raise
            except Exception:
                pass

        if not validate_model_files(model_dir):
            raise HTTPException(status_code=400, detail="Model files are incomplete or missing")

        run_id = f"predict_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        prediction_dir = PREDICTIONS_DIR / run_id
        prediction_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = prediction_dir / raw_fasta.filename
        if not save_uploaded_file(raw_fasta, fasta_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

        params = {
            "cluster_method": cluster_method,
            "kmeans_n": kmeans_n,
            "dbscan_eps": dbscan_eps,
            "dbscan_min": dbscan_min,
            "threshold": threshold
        }

        background_tasks.add_task(run_prediction_task, run_id, chosen_model, str(fasta_path), params)

        return {"message": "Prediction job started successfully", "run_id": run_id, "model_run_id": chosen_model, "parameters": params, "status_endpoint": f"/runs/{run_id}"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/runs/{run_id}")
def get_run_status(run_id: str, request: Request):
    try:
        if run_id.startswith("predict"):
            run_dir = PREDICTIONS_DIR / run_id
        elif run_id.startswith("train"):
            run_dir = MODELS_DIR / run_id
        else:
            raise HTTPException(status_code=400, detail="Invalid run_id format")

        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run ID not found")

        status_file = run_dir / "status.json"
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="Status file for this run not found")

        with status_file.open("r", encoding="utf-8") as f:
            job_details = json.load(f)

        if job_details.get("start_time") and job_details.get("end_time"):
            st = _to_number_maybe(job_details.get("start_time"))
            ed = _to_number_maybe(job_details.get("end_time"))
            if st is not None and ed is not None:
                job_details["runtime_seconds"] = ed - st
            else:
                job_details["runtime_seconds"] = None
        elif job_details.get("start_time"):
            st = _to_number_maybe(job_details.get("start_time"))
            job_details["current_runtime_seconds"] = (time.time() - st) if st is not None else None

        results = {}
        try:
            files = [p.name for p in run_dir.iterdir() if p.is_file()]
            results["files"] = files
        except Exception:
            results["files_error"] = "could not list files"

        if job_details.get("status") == "completed":
            if run_id.startswith("predict"):
                medoid_file = run_dir / "cluster_medoid_predictions.csv"
                if medoid_file.exists():
                    try:
                        import pandas as _pd
                        df_med = _pd.read_csv(medoid_file)
                        results["medoid_predictions"] = df_med.to_dict(orient="records")
                    except Exception:
                        try:
                            rows = []
                            with medoid_file.open("r", newline="", encoding="utf-8") as f:
                                reader = _csv.DictReader(f)
                                for r in reader:
                                    rows.append(r)
                            results["medoid_predictions"] = rows
                        except Exception as e:
                            results["medoid_predictions_error"] = f"Could not read medoid CSV: {e}"

                abundance_file = run_dir / "species_abundance_by_reads.csv"
                if abundance_file.exists():
                    try:
                        import pandas as _pd
                        df_ab = _pd.read_csv(abundance_file)
                        results["species_abundance_by_reads"] = df_ab.to_dict(orient="records")
                    except Exception:
                        pass

                visuals = {}
                for img in ["cluster_scatter.png", "species_abundance_bar.png", "species_composition_pie.png"]:
                    p = run_dir / img
                    if p.exists():
                        try:
                            visuals[p.name] = str(request.url_for("get_run_file", run_id=run_id, filename=p.name))
                        except Exception:
                            base = str(request.base_url).rstrip("/")
                            visuals[p.name] = f"{base}/runs/{run_id}/file/{quote(p.name, safe='')}"

                for p in run_dir.glob("*.png"):
                    if p.name not in visuals:
                        try:
                            visuals[p.name] = str(request.url_for("get_run_file", run_id=run_id, filename=p.name))
                        except Exception:
                            base = str(request.base_url).rstrip("/")
                            visuals[p.name] = f"{base}/runs/{run_id}/file/{quote(p.name, safe='')}"

                results["visualizations"] = visuals

            results["download_available"] = True

        return {"job_details": job_details, "results": results, "download_endpoint": f"/runs/{run_id}/download" if job_details.get("status") == "completed" else None}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Status endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/models")
def get_models():
    try:
        completed_models = []

        for model_dir in MODELS_DIR.iterdir():
            if not model_dir.is_dir():
                continue

            status_file = model_dir / "status.json"
            if not status_file.exists():
                continue

            try:
                with status_file.open("r", encoding="utf-8") as f:
                    status_data = json.load(f)

                if status_data.get("status") == "completed":
                    raw_end = status_data.get("end_time")
                    raw_start = status_data.get("start_time")
                    end_num = _to_number_maybe(raw_end)
                    start_num = _to_number_maybe(raw_start)

                    formatted_time = "N/A"
                    if end_num is not None:
                        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_num))

                    runtime = None
                    if (start_num is not None) and (end_num is not None):
                        runtime = end_num - start_num

                    completed_models.append({
                        "run_id": status_data.get("run_id", model_dir.name),
                        "end_time": formatted_time,
                        "end_time_raw": end_num if end_num is not None else 0,
                        "runtime_seconds": runtime,
                        "parameters": status_data.get("parameters", {}),
                        "files_valid": validate_model_files(model_dir)
                    })
            except Exception as e:
                print(f"Error reading model status {model_dir.name}: {e}")
                continue

        completed_models.sort(key=lambda x: x.get("end_time_raw", 0) or 0, reverse=True)

        for m in completed_models:
            if "end_time_raw" in m:
                m.pop("end_time_raw", None)

        return {"models": completed_models, "total_count": len(completed_models)}
    except Exception as e:
        print(f"Models endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/runs/{run_id}/file/{filename}")
def get_run_file(run_id: str, filename: str):
    safe_name = Path(filename).name
    if run_id.startswith("predict"):
        run_dir = PREDICTIONS_DIR / run_id
    elif run_id.startswith("train"):
        run_dir = MODELS_DIR / run_id
    else:
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    file_path = run_dir / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = None
    if file_path.suffix.lower() == ".png":
        media_type = "image/png"
    elif file_path.suffix.lower() in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif file_path.suffix.lower() == ".csv":
        media_type = "text/csv"

    return FileResponse(str(file_path), media_type=media_type, filename=safe_name)

@app.get("/runs/{run_id}/medoid")
def get_medoid_csv(run_id: str):
    try:
        if run_id.startswith("predict"):
            run_dir = PREDICTIONS_DIR / run_id
        elif run_id.startswith("train"):
            run_dir = MODELS_DIR / run_id
        else:
            raise HTTPException(status_code=400, detail="Invalid run_id format")

        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run ID not found")

        medoid_path = run_dir / "cluster_medoid_predictions.csv"
        if medoid_path.exists():
            return FileResponse(str(medoid_path), media_type="text/csv", filename=medoid_path.name)

        for zpath in run_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    for name in zf.namelist():
                        if name.lower().endswith("cluster_medoid_predictions.csv") or ("medoid" in name.lower() and name.lower().endswith(".csv")):
                            data = zf.read(name)
                            out_name = Path(name).name
                            return StreamingResponse(io.BytesIO(data), media_type="text/csv",
                                                     headers={"Content-Disposition": f'attachment; filename="{out_name}"'})
            except Exception as e:
                print(f"Could not read zip {zpath}: {e}")
                continue

        status_file = run_dir / "status.json"
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="No medoid CSV on disk and no status available to generate it")

        try:
            with status_file.open("r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read status file: {e}")

        results_block = status_data.get("results") if isinstance(status_data.get("results"), dict) else {}
        medoid_list = results_block.get("medoid_predictions") or status_data.get("medoid_predictions")
        if not isinstance(medoid_list, list) or len(medoid_list) == 0:
            raise HTTPException(status_code=404, detail="No medoid predictions available to generate CSV")

        fieldnames = []
        for r in medoid_list:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)

        if not fieldnames:
            fieldnames = list(medoid_list[0].keys()) if isinstance(medoid_list[0], dict) else []

        csv_buf = io.StringIO()
        writer = _csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in medoid_list:
            if isinstance(row, dict):
                normalized = {k: ("" if row.get(k) is None else (row.get(k) if isinstance(row.get(k), (str, int, float, bool)) else json.dumps(row.get(k)))) for k in fieldnames}
                writer.writerow(normalized)
            else:
                writer.writerow({"value": json.dumps(row)})

        csv_bytes = csv_buf.getvalue().encode("utf-8")
        out_name = f"{run_id}_cluster_medoid_predictions.csv"
        return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{out_name}"'})

    except HTTPException:
        raise
    except Exception as e:
        print(f"get_medoid_csv error for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/runs/{run_id}/medoid/json")
def get_medoid_json(run_id: str):
    try:
        if run_id.startswith("predict"):
            run_dir = PREDICTIONS_DIR / run_id
        elif run_id.startswith("train"):
            run_dir = MODELS_DIR / run_id
        else:
            raise HTTPException(status_code=400, detail="Invalid run_id format")

        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run ID not found")

        medoid_path = run_dir / "cluster_medoid_predictions.csv"
        if medoid_path.exists():
            try:
                import pandas as _pd
                df_med = _pd.read_csv(medoid_path)
                return JSONResponse(content={"medoid_predictions": df_med.to_dict(orient="records")})
            except Exception:
                try:
                    rows = []
                    with medoid_path.open("r", newline="", encoding="utf-8") as f:
                        reader = _csv.DictReader(f)
                        for r in reader:
                            rows.append(r)
                    return JSONResponse(content={"medoid_predictions": rows})
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Could not parse medoid CSV: {e}")

        for zpath in run_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    for name in zf.namelist():
                        if name.lower().endswith("cluster_medoid_predictions.csv") or ("medoid" in name.lower() and name.lower().endswith(".csv")):
                            raw = zf.read(name)
                            try:
                                import pandas as _pd
                                df_med = _pd.read_csv(io.BytesIO(raw))
                                return JSONResponse(content={"medoid_predictions": df_med.to_dict(orient="records")})
                            except Exception:
                                try:
                                    rows = []
                                    txt = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8")
                                    reader = _csv.DictReader(txt)
                                    for r in reader:
                                        rows.append(r)
                                    return JSONResponse(content={"medoid_predictions": rows})
                                except Exception as e:
                                    raise HTTPException(status_code=500, detail=f"Could not parse medoid CSV from zip {zpath}: {e}")
            except Exception as e:
                print(f"Could not read zip {zpath}: {e}")
                continue

        status_file = run_dir / "status.json"
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="No medoid file on disk and no status available")

        try:
            with status_file.open("r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read status file: {e}")

        results_block = status_data.get("results") if isinstance(status_data.get("results"), dict) else {}
        medoid_list = results_block.get("medoid_predictions") or status_data.get("medoid_predictions")
        if not isinstance(medoid_list, list):
            raise HTTPException(status_code=404, detail="No medoid predictions available in status")

        return JSONResponse(content={"medoid_predictions": medoid_list})
    except HTTPException:
        raise
    except Exception as e:
        print(f"get_medoid_json error for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/runs/{run_id}/download")
def download_run_files(run_id: str):
    try:
        if run_id.startswith("predict"):
            run_dir = PREDICTIONS_DIR / run_id
        elif run_id.startswith("train"):
            run_dir = MODELS_DIR / run_id
        else:
            raise HTTPException(status_code=400, detail="Invalid run_id format")

        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run ID not found")

        status_file = run_dir / "status.json"
        if status_file.exists():
            with status_file.open("r", encoding="utf-8") as f:
                status_data = json.load(f)
            if status_data.get("status") != "completed":
                raise HTTPException(status_code=400, detail="Run is not completed yet")
        else:
            status_data = {}

        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            added_any = False

            medoid_path = run_dir / "cluster_medoid_predictions.csv"
            if medoid_path.exists():
                try:
                    zf.write(medoid_path, medoid_path.name)
                    added_any = True
                except Exception as e:
                    print(f"Error adding medoid CSV from disk: {e}")
            else:
                results_block = status_data.get("results") if isinstance(status_data.get("results"), dict) else {}
                medoid_list = results_block.get("medoid_predictions") or status_data.get("medoid_predictions")
                if isinstance(medoid_list, list) and len(medoid_list) > 0:
                    try:
                        fieldnames = []
                        for r in medoid_list:
                            if isinstance(r, dict):
                                for k in r.keys():
                                    if k not in fieldnames:
                                        fieldnames.append(k)
                        if not fieldnames:
                            fieldnames = list(medoid_list[0].keys())
                        csv_buf = io.StringIO()
                        writer = _csv.DictWriter(csv_buf, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in medoid_list:
                            normalized = {k: (json.dumps(v) if not isinstance(v, (str, int, float, bool, type(None))) else ("" if v is None else v)) for k, v in (row.items() if isinstance(row, dict) else [])}
                            writer.writerow(normalized)
                        zf.writestr("cluster_medoid_predictions.csv", csv_buf.getvalue())
                        added_any = True
                    except Exception as e:
                        print(f"Could not generate medoid CSV from status data: {e}")

            for csv_name in ["species_abundance_by_reads.csv", "sequence_cluster_assignments.csv"]:
                p = run_dir / csv_name
                if p.exists():
                    try:
                        zf.write(p, p.name)
                        added_any = True
                    except Exception as e:
                        print(f"Error adding {csv_name}: {e}")

            image_patterns = ["*.png", "*.jpg", "*.jpeg"]
            for pat in image_patterns:
                for p in run_dir.glob(pat):
                    try:
                        zf.write(p, p.name)
                        added_any = True
                    except Exception as e:
                        print(f"Error adding image {p.name}: {e}")

            if status_file.exists():
                try:
                    with status_file.open("r", encoding="utf-8") as f:
                        stat = json.load(f)
                    copy_stat = dict(stat)
                    if "traceback" in copy_stat:
                        copy_stat["traceback"] = "<traceback omitted in zip>"
                    zf.writestr("status.json", json.dumps(copy_stat, indent=2))
                    added_any = True
                except Exception:
                    try:
                        zf.write(status_file, "status.json")
                        added_any = True
                    except Exception as e:
                        print(f"Error adding status.json: {e}")

            try:
                summary_rows = []
                job_type = status_data.get("job_type") or ("prediction" if run_id.startswith("predict") else "train")
                summary = {
                    "run_id": run_id,
                    "job_type": job_type,
                    "status": status_data.get("status", ""),
                    "model_run_id": (status_data.get("parameters") or {}).get("model_run_id") or "",
                    "filename": (status_data.get("parameters") or {}).get("filename") or "",
                    "start_time": status_data.get("start_time"),
                    "end_time": status_data.get("end_time")
                }
                results_info = status_data.get("results_info") or {}
                summary.update({
                    "total_sequences": results_info.get("total_sequences") if results_info else "",
                    "num_clusters": results_info.get("num_clusters") if results_info else "",
                    "medoids_found": results_info.get("medoids_found") if results_info else ""
                })
                try:
                    st = _to_number_maybe(status_data.get("start_time"))
                    ed = _to_number_maybe(status_data.get("end_time"))
                    summary["runtime_seconds"] = (ed - st) if (st is not None and ed is not None) else ""
                except Exception:
                    summary["runtime_seconds"] = ""
                summary_rows.append(summary)

                if summary_rows:
                    fieldnames = list(summary_rows[0].keys())
                    buf = io.StringIO()
                    writer = _csv.DictWriter(buf, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in summary_rows:
                        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
                    zf.writestr("summary.csv", buf.getvalue())
                    added_any = True
            except Exception as e:
                print(f"Could not write summary.csv: {e}")

            try:
                readme_lines = [
                    "Results ZIP for run: " + run_id,
                    "",
                    "Files included (when present):",
                    "- cluster_medoid_predictions.csv  : medoid predictions per cluster (labels + confidences)",
                    "- sequence_cluster_assignments.csv: assignment of sequences to clusters",
                    "- species_abundance_by_reads.csv  : species abundance table by read counts",
                    "- *.png / *.jpg                    : visualization images (cluster scatter, abundance bar, composition pie)",
                    "- status.json                      : job status (traceback omitted)",
                    "- summary.csv                      : small friendly summary of run metadata",
                    "",
                    "If an expected CSV was not present on disk, the server attempted to generate it from the run status when possible.",
                    "",
                    "Note: if images are missing or appear as placeholders in the UI, check server logs and ensure the files exist in the run directory."
                ]
                zf.writestr("README.txt", "\n".join(readme_lines))
            except Exception as e:
                print(f"Could not write README.txt: {e}")

            if not added_any:
                raise HTTPException(status_code=404, detail="No files found to include in ZIP")

        mem_zip.seek(0)
        out_filename = f"{run_id}_results.zip"
        return StreamingResponse(
            io.BytesIO(mem_zip.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{out_filename}"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Download endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/debug/runs")
def debug_runs():
    return {
        "models_dir": str(MODELS_DIR),
        "predictions_dir": str(PREDICTIONS_DIR),
        "model_runs": [d.name for d in MODELS_DIR.iterdir() if d.is_dir()],
        "prediction_runs": [d.name for d in PREDICTIONS_DIR.iterdir() if d.is_dir()]
    }

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
