"""Diagnostic helper to inspect why all predictions are 'Unknown'.

It loads model artifacts from model_output, vectorizes `raw_sequences.fasta`,
computes reduced & scaled features, shows DBSCAN labels with current eps/min,
prints nearest-neighbor distance stats (to choose eps), and shows MLP
prediction probabilities for medoids and all points.

Run:
    python debug_cluster.py
"""
from pathlib import Path
import os, sys
import numpy as np
from collections import Counter
import joblib
from math import inf
from Bio import SeqIO
import importlib.util
import inspect

ROOT = Path(__file__).parent
OUT = ROOT / "model_output"

def read_seqs(fasta):
    ids, seqs = [], []
    for rec in SeqIO.parse(fasta, "fasta"):
        ids.append(rec.id)
        seqs.append(str(rec.seq).upper().replace('U','T'))
    return ids, seqs

def pairwise_nearest_distances(X):
    # brute force but fine for small N
    N = X.shape[0]
    dists = np.full(N, np.inf)
    for i in range(N):
        diffs = X - X[i:i+1]
        dd = np.linalg.norm(diffs, axis=1)
        dd[i] = np.inf
        dists[i] = dd.min()
    return dists

def main():
    fasta = ROOT / "raw_sequences.fasta"
    if not fasta.exists():
        print("raw_sequences.fasta not found")
        sys.exit(1)

    ids, seqs = read_seqs(str(fasta))
    print(f"Loaded {len(seqs)} sequences from {fasta.name}")

    # Ensure local classes (e.g. KmerVectorizer) are importable under __main__
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

    # load artifacts
    vec = joblib.load(OUT / "kmer_vectorizer.joblib")
    reducer = joblib.load(OUT / "reducer.joblib")
    scaler = joblib.load(OUT / "scaler.joblib")
    mlp = joblib.load(OUT / "mlp_model.joblib")
    le = joblib.load(OUT / "label_encoder.joblib")

    print("Artifacts loaded:")
    print(" - vectorizer k:", getattr(vec, 'k', None))
    print(" - reducer n_components:", getattr(reducer, 'n_components', None))
    print(" - scaler:", type(scaler))
    print(" - mlp classes:", getattr(mlp, 'classes_', None))
    print(" - label encoder classes:", getattr(le, 'classes_', None))

    # transform
    X = vec.transform(seqs)
    print("X shape:", X.shape)
    Xr = reducer.transform(X)
    print("Reduced Xr shape:", Xr.shape)
    Xr_s = scaler.transform(Xr)
    print("Scaled reduced shape:", Xr_s.shape)
    print("Scaled stats: mean", np.mean(Xr_s), "std", np.std(Xr_s), "min", np.min(Xr_s), "max", np.max(Xr_s))

    # nearest-neighbor distances
    dists = pairwise_nearest_distances(Xr_s)
    print("Nearest-neighbor distances: min", float(np.min(dists)), "median", float(np.median(dists)), "mean", float(np.mean(dists)), "max", float(np.max(dists)))

    # run DBSCAN with the configured eps/min (try to read them from CLI args?)
    from sklearn.cluster import DBSCAN
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--min', type=int, default=3)
    args = parser.parse_args()

    db = DBSCAN(eps=args.eps, min_samples=args.min, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(Xr_s)
    print(f"DBSCAN (eps={args.eps}, min_samples={args.min}) labels unique:", sorted(set(labels)))
    print(Counter(labels))

    # medoids
    medoids = {}
    unique = [l for l in set(labels) if l != -1]
    for lab in unique:
        idxs = np.where(labels == lab)[0]
        centroid = Xr_s[idxs].mean(axis=0)
        d = np.linalg.norm(Xr_s[idxs] - centroid, axis=1)
        medoids[lab] = idxs[np.argmin(d)]
    print("Medoids found:", medoids)

    if medoids:
        for cl,midx in medoids.items():
            x = vec.transform([seqs[midx]])
            xr = reducer.transform(x)
            xr_s = scaler.transform(xr)
            probs = mlp.predict_proba(xr_s)[0]
            maxp = float(np.max(probs))
            pred_numeric = mlp.classes_[int(np.argmax(probs))]
            pred_label = le.inverse_transform([pred_numeric])[0]
            print(f"medoid cluster {cl} index {midx} pred={pred_label} maxp={maxp} probs(top5)={np.sort(probs)[-5:][::-1]} ")
    else:
        print("No medoids (no clusters apart from noise). This explains 'Unknown' predictions.")

    # show mlp max probs for all points (even if no medoids) to check classifier confidence
    try:
        all_probs = mlp.predict_proba(Xr_s)
        maxps = np.max(all_probs, axis=1)
        print("MLP max-prob stats: min", float(np.min(maxps)), "median", float(np.median(maxps)), "mean", float(np.mean(maxps)), "max", float(np.max(maxps)))
    except Exception as e:
        print("Could not get predict_proba for all points:", e)

    # try DBSCAN with larger eps to see if clusters appear
    for eps in [0.6, 0.8, 1.0, 1.5]:
        db2 = DBSCAN(eps=eps, min_samples=args.min, metric='euclidean', n_jobs=-1)
        labs2 = db2.fit_predict(Xr_s)
        print(f"eps={eps} unique:", sorted(set(labs2)), "counts:", Counter(labs2))

if __name__ == '__main__':
    main()
