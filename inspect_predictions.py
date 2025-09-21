from pathlib import Path
from joblib import load
from Bio import SeqIO
import numpy as np
import importlib.util
import sys, inspect

ROOT = Path(__file__).parent
OUT = ROOT / "model_output"

def read_seqs(fasta):
    ids, seqs = [], []
    for rec in SeqIO.parse(fasta, "fasta"):
        ids.append(rec.id)
        seqs.append(str(rec.seq).upper().replace('U','T'))
    return ids, seqs

def topk_probs(probs, classes, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in idx]

def main():
    fasta = ROOT / "raw_sequences.fasta"
    if not fasta.exists():
        print("raw_sequences.fasta not found")
        return

    ids, seqs = read_seqs(str(fasta))
    print(f"Loaded {len(seqs)} sequences from {fasta.name}")

    # Ensure custom classes from the pipeline module are available for unpickling
    try:
        repo_module_name = "clust_nn_pipeline_sklearn"
        repo_path = ROOT / (repo_module_name + ".py")
        if repo_path.exists():
            spec = importlib.util.spec_from_file_location(repo_module_name, str(repo_path))
            repo_mod = importlib.util.module_from_spec(spec)
            sys.modules[repo_module_name] = repo_mod
            spec.loader.exec_module(repo_mod)
            main_mod = sys.modules.get("__main__")
            for name, obj in inspect.getmembers(repo_mod, inspect.isclass):
                if not hasattr(main_mod, name):
                    setattr(main_mod, name, obj)
    except Exception:
        pass

    vec = load(OUT / "kmer_vectorizer.joblib")
    reducer = load(OUT / "reducer.joblib")
    scaler = load(OUT / "scaler.joblib")
    mlp = load(OUT / "mlp_model.joblib")
    le = load(OUT / "label_encoder.joblib")

    X = vec.transform(seqs)
    Xr = reducer.transform(X)
    Xr_s = scaler.transform(Xr)

    try:
        probs = mlp.predict_proba(Xr_s)
    except Exception as e:
        print("Model does not support predict_proba:", e)
        preds = mlp.predict(Xr_s)
        labels = le.inverse_transform(preds)
        for sid, lab in zip(ids, labels):
            print(sid, lab)
        return

    classes = mlp.classes_
    class_names = le.inverse_transform(classes)

    for i, sid in enumerate(ids):
        top = topk_probs(probs[i], classes, k=min(3, len(classes)))
        top_named = [(class_names[list(classes).index(c)], p) for c,p in top]
        print(f"{sid}: top1={top_named[0][0]} ({top_named[0][1]:.6f}) | top3={top_named}")

if __name__ == '__main__':
    main()
