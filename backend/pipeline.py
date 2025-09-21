#!/usr/bin/env python3
import os, sys, argparse, glob, re, json
from collections import Counter
import itertools
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO

# sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

# viz
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Helpers
# -------------------------
def ensure_fasta_headers(in_file, tmp_file="tmp_with_headers.fasta"):
    with open(in_file, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    has_header = any(l.startswith(">") for l in lines)
    if has_header:
        return in_file
    with open(tmp_file, "w") as out:
        idx = 1
        for line in lines:
            if not line:
                continue
            out.write(f">seq{idx}\n{line.strip()}\n")
            idx += 1
    return tmp_file

def read_fasta_to_list(fasta_path):
    seqs = []
    ids = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).upper().replace("U", "T")
        if seq:
            seqs.append(seq)
            ids.append(rec.id)
    return ids, seqs

# -------------------------
# K-mer vectorizer
# -------------------------
class KmerVectorizer:
    def __init__(self, k=4):
        self.k = int(k)
        self.vocab = [''.join(p) for p in itertools.product('ATGC', repeat=self.k)]
        self.vocab_size = len(self.vocab)

    def transform_sequence(self, seq):
        seq = seq.upper()
        counts = Counter()
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            if 'N' in kmer or any(ch not in 'ATGC' for ch in kmer):
                continue
            counts[kmer] += 1
        total = max(sum(counts.values()), 1)
        return np.array([counts[k]/total for k in self.vocab], dtype=np.float32)

    def transform(self, sequences):
        X = np.zeros((len(sequences), self.vocab_size), dtype=np.float32)
        for i, seq in enumerate(tqdm(sequences, desc="Vectorizing (k-mer)")):
            X[i,:] = self.transform_sequence(seq)
        return X

# -------------------------
# Dimensionality reduction
# -------------------------
def make_reducer(X, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xr = svd.fit_transform(X)
    return svd, Xr

# -------------------------
# Clustering
# -------------------------
def cluster_dbscan(Xr, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(Xr)
    return labels

def cluster_kmeans(Xr, n_clusters=10):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xr)
    return labels

# -------------------------
# Medoid selection
# -------------------------
def cluster_medoid_indices(X, labels):
    medoids = {}
    unique = [l for l in set(labels) if l != -1]
    for lab in unique:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            continue
        centroid = X[idxs].mean(axis=0)
        dists = np.linalg.norm(X[idxs] - centroid, axis=1)
        med_idx = idxs[np.argmin(dists)]
        medoids[lab] = med_idx
    return medoids

# -------------------------
# sklearn MLP training
# -------------------------
def train_mlp_sklearn(X, y, epochs=50):
    mlp = MLPClassifier(hidden_layer_sizes=(512,256,128),
                        activation='relu',
                        solver='adam',
                        alpha=1e-4,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=epochs,
                        verbose=True,
                        random_state=42)
    mlp.fit(X, y)
    return mlp

# -------------------------
# Pipeline functions
# -------------------------
def train_from_labeled_fasta(train_fasta, outdir, k=4, pca_comp=50, epochs=50, centroid_percentile=95.0):
    os.makedirs(outdir, exist_ok=True)
    print(f"[{outdir}] Starting training...")
    fasta = ensure_fasta_headers(train_fasta)
    ids, seqs = read_fasta_to_list(fasta)
    labels = [re.sub(r'[^A-Za-z0-9_]', '_', h.split()[0]) for h in ids]
    
    vec = KmerVectorizer(k=k)
    X = vec.transform(seqs)
    
    reducer, Xr = make_reducer(X, n_components=min(pca_comp, X.shape[1]-1))
    
    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)
    
    le = LabelEncoder()
    y = le.fit_transform(labels)

    counts = Counter(y)
    if any(c < 2 for c in counts.values()):
        print("Warning: Some classes have < 2 samples and will be removed.")
        keep_mask = np.isin(y, [label for label, c in counts.items() if c >= 2])
        Xr_s, y = Xr_s[keep_mask], y[keep_mask]
        labels_kept = [labels[i] for i, keep in enumerate(keep_mask) if keep]
        le = LabelEncoder().fit(labels_kept)
        y = le.transform(labels_kept)

    X_train, X_test, y_train, y_test = train_test_split(Xr_s, y, test_size=0.2, random_state=42, stratify=y)
    
    mlp = train_mlp_sklearn(X_train, y_train, epochs=epochs)
    
    print("Classification report on test set:")
    print(classification_report(y_test, mlp.predict(X_test), zero_division=0, target_names=le.inverse_transform(np.unique(y_test))))
    
    # save artifacts
    joblib.dump(vec, os.path.join(outdir, "kmer_vectorizer.joblib"))
    joblib.dump(reducer, os.path.join(outdir, "reducer.joblib"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.joblib"))
    joblib.dump(le, os.path.join(outdir, "label_encoder.joblib"))
    joblib.dump(mlp, os.path.join(outdir, "mlp_model.joblib"))
    
    centroids, thresholds = {}, {}
    for class_idx, class_label in enumerate(le.classes_):
        mask = (y == class_idx)
        if not np.any(mask): continue
        Xc = Xr_s[mask]
        centroid = Xc.mean(axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        centroids[class_label] = centroid
        thresholds[class_label] = float(np.percentile(dists, float(centroid_percentile)))
    joblib.dump({'centroids': centroids, 'thresholds': thresholds}, os.path.join(outdir, 'class_centroids.joblib'))
    
    print(f"[{outdir}] Saved all model artifacts.")
    return True

def cluster_and_predict(raw_fasta, model_dir, outdir, k=4, pca_comp=50, cluster_method="dbscan", dbscan_eps=0.5, dbscan_min=5, kmeans_n=10, threshold=0.7):
    os.makedirs(outdir, exist_ok=True)
    print(f"[{outdir}] Starting prediction with model from [{model_dir}]...")
    fasta = ensure_fasta_headers(raw_fasta, tmp_file=os.path.join(outdir, "tmp_with_headers.fasta"))
    ids, seqs = read_fasta_to_list(fasta)

    # Load artifacts from the specified model directory
    vec = joblib.load(os.path.join(model_dir, "kmer_vectorizer.joblib"))
    reducer = joblib.load(os.path.join(model_dir, "reducer.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    mlp = joblib.load(os.path.join(model_dir, "mlp_model.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    centroids_data = joblib.load(os.path.join(model_dir, "class_centroids.joblib"))

    X = vec.transform(seqs)
    Xr = reducer.transform(X)
    Xr_s = scaler.transform(Xr)
    
    if cluster_method.lower() == "dbscan":
        labels = cluster_dbscan(Xr_s, eps=dbscan_eps, min_samples=dbscan_min)
    else:
        labels = cluster_kmeans(Xr_s, n_clusters=kmeans_n)
    
    print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}, Noise points: {np.sum(labels == -1)}")
    
    medoids = cluster_medoid_indices(Xr_s, labels)
    medoid_rows = []
    for cl, midx in medoids.items():
        xr_s_medoid = Xr_s[midx].reshape(1, -1)
        probs = mlp.predict_proba(xr_s_medoid)[0]
        max_prob = np.max(probs)
        pred_label_idx = np.argmax(probs)
        pred_label = le.inverse_transform([mlp.classes_[pred_label_idx]])[0] if max_prob >= threshold else "Unknown"
        
        if pred_label != "Unknown" and pred_label in centroids_data['centroids']:
            dist = np.linalg.norm(xr_s_medoid - centroids_data['centroids'][pred_label])
            if dist > centroids_data['thresholds'][pred_label]:
                pred_label = "Unknown"
        
        medoid_rows.append((cl, midx, ids[midx], pred_label, float(max_prob)))

    medoid_df = pd.DataFrame(medoid_rows, columns=["cluster", "medoid_index", "medoid_id", "predicted_species", "confidence"])
    medoid_df.to_csv(os.path.join(outdir, "cluster_medoid_predictions.csv"), index=False)

    assignment_map = medoid_df.set_index('cluster')['predicted_species'].to_dict()
    assign_df = pd.DataFrame({
        "sequence_id": ids,
        "cluster": labels,
        "predicted_species": [assignment_map.get(l, "Unknown") for l in labels]
    })
    assign_df.to_csv(os.path.join(outdir, "sequence_cluster_assignments.csv"), index=False)
    print(f"[{outdir}] Saved sequence assignments and medoid predictions.")

    # Generate final report and plots
    try:
        species_counts = assign_df['predicted_species'].value_counts().reset_index()
        species_counts.columns = ['predicted_species', 'n_reads']
        species_counts['percentage'] = (100 * species_counts['n_reads'] / len(assign_df)).round(2)
        species_counts.to_csv(os.path.join(outdir, "species_abundance.csv"), index=False)
        
        # Create a JSON report
        report = {
            "total_sequences": len(assign_df),
            "clusters_found": len(medoids),
            "unclustered_sequences": int(np.sum(labels == -1)),
            "composition": species_counts.to_dict(orient='records')
        }
        with open(os.path.join(outdir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=4)
            
        # Bar plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x="percentage", y="predicted_species", data=species_counts.head(20))
        plt.title("Top 20 Species Abundance")
        plt.xlabel("Percentage of Reads (%)")
        plt.ylabel("Predicted Species")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "species_abundance_bar.png"))
        plt.close()

        # Scatter plot
        plt.figure(figsize=(10, 8))
        unique_labels = sorted(set(labels))
        palette = sns.color_palette("hsv", len(unique_labels))
        for i, label in enumerate(unique_labels):
            if label == -1:
                plt.scatter(Xr[labels==label, 0], Xr[labels==label, 1], s=5, color='gray', label='Noise')
            else:
                plt.scatter(Xr[labels==label, 0], Xr[labels==label, 1], s=10, color=palette[i], label=f'Cluster {label}')
        plt.title("Sequence Clusters (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "cluster_scatter.png"))
        plt.close()
        print(f"[{outdir}] Saved report.json and plots.")

    except Exception as e:
        print(f"Could not generate final reports or plots: {e}")

    return True

