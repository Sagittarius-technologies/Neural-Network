import os
import io
import uuid
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any
import sys

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Path Correction for Module Import ---
# Add the directory containing this script to the Python path.
# This ensures that 'pipeline.py' can be imported regardless of where
# the 'uvicorn' command is run from.
sys.path.append(str(Path(__file__).resolve().parent))

# Import the refactored pipeline logic.
# This assumes 'pipeline.py' is in the same 'backend/' directory.
import backend.pipeline as pipeline

# --- Configuration & Setup ---
ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "runs"
MODELS_DIR = RUNS_DIR / "models"
PREDICTIONS_DIR = RUNS_DIR / "predictions"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

app = FastAPI(
    title="eDNA Clustering & NN API",
    description="A robust API for training models and predicting species from eDNA sequences.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API Data Validation ---
class Job(BaseModel):
    """Represents a training or prediction job's metadata."""
    run_id: str
    status: str = "queued"
    start_time: str = ""
    end_time: str = ""
    error: str = ""
    parameters: Dict[str, Any]

class TrainJob(Job):
    job_type: str = "training"

class PredictJob(Job):
    job_type: str = "prediction"
    model_run_id: str

# --- Job Management ---
def write_job_status(run_dir: Path, job_data: Job):
    """Writes the current job status to a status.json file."""
    with open(run_dir / "status.json", "w") as f:
        f.write(job_data.model_dump_json(indent=4))

@contextmanager
def job_runner(run_dir: Path, job: Job):
    """A context manager to handle job status updates and error logging."""
    job.status = "running"
    job.start_time = datetime.utcnow().isoformat() + "Z"
    write_job_status(run_dir, job)
    try:
        yield
        job.status = "completed"
    except Exception as e:
        job.status = "failed"
        job.error = f"{type(e).__name__}: {str(e)}"
        print(f"Job {job.run_id} failed: {e}") # Log to server console
    finally:
        job.end_time = datetime.utcnow().isoformat() + "Z"
        write_job_status(run_dir, job)

# --- Background Task Functions ---
def run_training_task(run_id: str, fasta_path: str, params: dict):
    """Background task wrapper for the training pipeline."""
    run_dir = MODELS_DIR / run_id
    job = TrainJob(run_id=run_id, parameters=params)
    with job_runner(run_dir, job):
        # *** This is the key part: we call the function from pipeline.py ***
        pipeline.train_from_labeled_fasta(
            train_fasta=fasta_path,
            outdir=str(run_dir),
            k=params['k'],
            pca_comp=params['pca_comp'],
            epochs=params['epochs'],
            centroid_percentile=params['centroid_percentile']
        )

def run_prediction_task(run_id: str, model_run_id: str, fasta_path: str, params: dict):
    """Background task wrapper for the prediction pipeline."""
    run_dir = PREDICTIONS_DIR / run_id
    model_dir = MODELS_DIR / model_run_id
    job = PredictJob(run_id=run_id, model_run_id=model_run_id, parameters=params)
    with job_runner(run_dir, job):
        # *** Here we call the prediction function from pipeline.py ***
        pipeline.cluster_and_predict(
            raw_fasta=fasta_path,
            model_dir=str(model_dir),
            outdir=str(run_dir),
            k=params['k'],
            pca_comp=params['pca_comp'],
            cluster_method=params['cluster_method'],
            dbscan_eps=params['dbscan_eps'],
            dbscan_min=params['dbscan_min'],
            kmeans_n=params['kmeans_n'],
            threshold=params['threshold']
        )

# --- API Endpoints ---
@app.post("/train", summary="Start a new model training job")
def train_endpoint(
    background_tasks: BackgroundTasks,
    fasta: UploadFile = File(..., description="Labeled FASTA file for training."),
    k: int = Form(4, description="k-mer size."),
    pca_comp: int = Form(50, description="Number of PCA components."),
    epochs: int = Form(50, description="Number of training epochs."),
    centroid_percentile: float = Form(95.0, description="Percentile for centroid distance threshold.")
):
    """Upload a labeled FASTA file to train a new model. This starts a background job."""
    run_id = f"train_{uuid.uuid4()}"
    run_dir = MODELS_DIR / run_id
    os.makedirs(run_dir)
    
    fasta_path = run_dir / "input.fasta"
    with open(fasta_path, "wb") as buffer:
        shutil.copyfileobj(fasta.file, buffer)

    params = {'k': k, 'pca_comp': pca_comp, 'epochs': epochs, 'centroid_percentile': centroid_percentile, 'filename': fasta.filename}
    job = TrainJob(run_id=run_id, parameters=params)
    write_job_status(run_dir, job)

    background_tasks.add_task(run_training_task, run_id, str(fasta_path), params)
    return {"message": "Training job started successfully.", "run_id": run_id}

@app.post("/predict", summary="Start a new prediction job")
def predict_endpoint(
    background_tasks: BackgroundTasks,
    model_run_id: str = Form(..., description="The run_id of the trained model to use."),
    raw_fasta: UploadFile = File(..., description="Raw FASTA file for prediction."),
    cluster_method: str = Form("kmeans", description="Clustering method: 'kmeans' or 'dbscan'."),
    kmeans_n: int = Form(10, description="Number of clusters for KMeans."),
    threshold: float = Form(0.7, description="Confidence threshold for prediction.")
):
    """Upload a raw FASTA file to classify sequences using a trained model."""
    if not (MODELS_DIR / model_run_id).exists():
        raise HTTPException(status_code=404, detail=f"Model run ID '{model_run_id}' not found.")
        
    run_id = f"predict_{uuid.uuid4()}"
    run_dir = PREDICTIONS_DIR / run_id
    os.makedirs(run_dir)

    fasta_path = run_dir / "input.fasta"
    with open(fasta_path, "wb") as buffer:
        shutil.copyfileobj(raw_fasta.file, buffer)

    try:
        with open(MODELS_DIR / model_run_id / "status.json", "r") as f:
            train_params = json.load(f)['parameters']
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail="Could not find training parameters for the specified model.")

    params = {
        'k': train_params.get('k', 4), 'pca_comp': train_params.get('pca_comp', 50),
        'cluster_method': cluster_method, 'kmeans_n': kmeans_n, 'threshold': threshold,
        'dbscan_eps': 0.5, 'dbscan_min': 5, # Default values for dbscan
        'filename': raw_fasta.filename
    }
    job = PredictJob(run_id=run_id, model_run_id=model_run_id, parameters=params)
    write_job_status(run_dir, job)

    background_tasks.add_task(run_prediction_task, run_id, model_run_id, str(fasta_path), params)
    return {"message": "Prediction job started successfully.", "run_id": run_id}

@app.get("/runs/{run_id}", summary="Get job status and results")
def get_run_details(run_id: str):
    """Fetches the status and results for a given training or prediction run_id."""
    run_dir = PREDICTIONS_DIR / run_id if run_id.startswith("predict") else MODELS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run ID not found.")

    response = {}
    if (status_path := run_dir / "status.json").exists():
        with open(status_path, "r") as f:
            response['job_details'] = json.load(f)
            
    if (report_path := run_dir / "report.json").exists():
        with open(report_path, "r") as f:
            response['results'] = json.load(f)
            
    if not response:
        raise HTTPException(status_code=404, detail="Status file not found for this run.")
        
    return response

@app.get("/models", summary="List all successfully trained models")
def list_trained_models():
    """Returns a list of all successfully trained models available for prediction."""
    models = []
    for run_dir in MODELS_DIR.iterdir():
        if run_dir.is_dir() and (status_path := run_dir / "status.json").exists():
            with open(status_path) as f:
                status = json.load(f)
                if status.get('status') == 'completed':
                    models.append(status)
    return sorted(models, key=lambda x: x.get('end_time', ''), reverse=True)

@app.get("/runs/{run_id}/download", summary="Download all results as a ZIP")
def download_run_results(run_id: str):
    """Downloads all output files for a given run_id as a single ZIP archive."""
    run_dir = PREDICTIONS_DIR / run_id if run_id.startswith("predict") else MODELS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run ID not found.")

    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as temp_zip:
        for f_path in run_dir.rglob('*'):
            if f_path.is_file():
                temp_zip.write(f_path, f_path.relative_to(run_dir))
    
    zip_io.seek(0)
    return StreamingResponse(
        zip_io, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={run_id}_results.zip"}
    )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the eDNA NN API. Navigate to /docs for the interactive API documentation."}

