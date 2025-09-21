"""Simple leave-one-species-out OOD test.

Usage: python ood_test.py --holdout Salmo_salar --epochs 20 --centroid_percentile 90
It will train on sequences_clean.fasta excluding the holdout species, then predict
on the held-out sequences and print prediction confidences and top labels.
"""
import argparse
from pathlib import Path
from Bio import SeqIO
import numpy as np
from collections import Counter
import joblib
import os

ROOT = Path(__file__).parent

def read_fasta_labels(fasta):
    ids, seqs, labels = [], [], []
    for rec in SeqIO.parse(fasta, 'fasta'):
        ids.append(rec.id)
        seqs.append(str(rec.seq).upper().replace('U','T'))
        labels.append(rec.id)  # in our cleaned FASTA labels are in id
    return ids, seqs, labels

def load_cleaned(fasta):
    ids, seqs, labels = [], [], []
    for rec in SeqIO.parse(fasta, 'fasta'):
        ids.append(rec.id)
        seqs.append(str(rec.seq).upper().replace('U','T'))
        labels.append(rec.id)
    return ids, seqs, labels

def parse_label_from_id(seq_id):
    # in sequences_clean.fasta, rec.id is the label (e.g., Salmo_salar)
    return seq_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--centroid_percentile', type=float, default=95.0,
                        help='Percentile to set per-class centroid distance threshold')
    args = parser.parse_args()

    cleaned = ROOT / 'sequences_clean.fasta'
    if not cleaned.exists():
        print('sequences_clean.fasta not found')
        return

    # Read cleaned FASTA and group by label
    labels = []
    seqs = []
    ids = []
    for rec in SeqIO.parse(str(cleaned), 'fasta'):
        ids.append(rec.id)
        seqs.append(str(rec.seq).upper().replace('U','T'))
        labels.append(rec.id)

    labels = np.array(labels)
    seqs = np.array(seqs)

    hold_mask = labels == args.holdout
    if hold_mask.sum() == 0:
        print('No sequences for holdout:', args.holdout)
        return

    train_seqs = seqs[~hold_mask]
    train_labels = labels[~hold_mask]
    test_seqs = seqs[hold_mask]
    test_ids = ids

    # quick temporary train using pipeline functions by importing module
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location('clust_nn_pipeline_sklearn', str(ROOT / 'clust_nn_pipeline_sklearn.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # vectorize and fit reducer/scaler/label encoder and MLP
    vec = mod.KmerVectorizer(k=4)
    X = vec.transform(list(train_seqs))
    reducer, Xr = mod.make_reducer(X, n_components=min(50, X.shape[1]-1))
    scaler = mod.StandardScaler()
    Xr_s = scaler.fit_transform(Xr)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(list(train_labels))

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(512,256,128), max_iter=args.epochs, random_state=42)
    mlp.fit(Xr_s, y)

    # prepare test features
    Xt = vec.transform(list(test_seqs))
    Xtr = reducer.transform(Xt)
    Xtr_s = scaler.transform(Xtr)

    probs = mlp.predict_proba(Xtr_s)
    maxps = probs.max(axis=1)
    preds = mlp.predict(Xtr_s)
    pred_labels = le.inverse_transform(preds)

    # compute per-class centroids and percentile thresholds on training embeddings
    centroids = {}
    thresholds = {}
    classes = le.classes_
    for ci, cls in enumerate(classes):
        mask = (y == ci)
        if mask.sum() == 0:
            continue
        class_X = Xr_s[mask]
        centroid = class_X.mean(axis=0)
        dists = np.linalg.norm(class_X - centroid, axis=1)
        thresh = float(np.percentile(dists, args.centroid_percentile))
        centroids[cls] = centroid
        thresholds[cls] = thresh

    print('Holdout species:', args.holdout)
    print('n_train:', len(train_seqs), 'n_test:', len(test_seqs))
    print('Test max-prob stats: min', float(maxps.min()), 'median', float(np.median(maxps)), 'mean', float(maxps.mean()))

    rejected = 0
    for i,p in enumerate(probs):
        top_idx = p.argsort()[::-1][:3]
        tops = [(le.inverse_transform([j])[0], float(p[j])) for j in top_idx]
        pred = pred_labels[i]
        x = Xtr_s[i]
        if pred in centroids:
            dist = float(np.linalg.norm(x - centroids[pred]))
            thresh = thresholds[pred]
            is_rejected = dist > thresh
        else:
            dist = float('nan')
            thresh = float('nan')
            is_rejected = True
        if is_rejected:
            rejected += 1
        print(f'TEST_{i}: top1={tops[0][0]} ({tops[0][1]:.6f}) top3={tops} dist={dist:.6f} thresh={thresh:.6f} rejected={is_rejected}')

    print(f'Rejected {rejected} / {len(test_seqs)} held-out samples (centroid_percentile={args.centroid_percentile})')

if __name__ == '__main__':
    main()
