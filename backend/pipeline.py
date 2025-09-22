#!/usr/bin/env python3
"""
Corrected pipeline.py (patched)

- Safe batch_size handling for small datasets
- KMeans n_init set to a safe integer (10)
- Top probabilities converted to readable labels+prob strings
- Explicit prints when files are saved (helps debug missing files)
- Minor robustness improvements
"""
import os
import re
import itertools
import joblib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Validation Functions
# -------------------------
def validate_fasta_file(fasta_path):
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"FASTA file is empty: {fasta_path}")
    return True

def validate_sequences(sequences):
    if not sequences:
        raise ValueError("No sequences found in FASTA file")
    valid_chars = set('ATGCNRYSWKMBDHVU')  # include ambiguous + U
    for i, seq in enumerate(sequences):
        if not seq:
            raise ValueError(f"Empty sequence found at position {i}")
        invalid = set(seq.upper()) - valid_chars
        if invalid:
            print(f"Warning: Sequence {i} contains unusual characters: {invalid}")
    return True

# -------------------------
# Helpers
# -------------------------
def ensure_fasta_headers(in_file, tmp_file=None):
    in_path = Path(in_file)
    if tmp_file is None:
        tmp_file = in_path.parent / f"tmp_with_headers_{in_path.stem}.fasta"
    if in_path.stat().st_size == 0:
        return str(in_path)
    with open(in_path, "r") as f:
        first = f.readline().strip()
    if first.startswith(">"):
        return str(in_path)
    with open(in_path, "r") as f_in, open(tmp_file, "w") as f_out:
        for i, line in enumerate(f_in):
            line = line.strip()
            if line:
                f_out.write(f">seq{i+1}\n{line}\n")
    return str(tmp_file)

def read_fasta_to_list(fasta_path):
    validate_fasta_file(fasta_path)
    seqs, ids = [], []
    temp_fasta_path = None
    try:
        processed_fasta_path = ensure_fasta_headers(fasta_path)
        if processed_fasta_path != fasta_path:
            temp_fasta_path = processed_fasta_path
        for rec in SeqIO.parse(processed_fasta_path, "fasta"):
            seq = str(rec.seq).upper().replace("U", "T")
            if seq:
                seqs.append(seq)
                ids.append(rec.id)
        if not seqs:
            raise ValueError("No valid sequences found in FASTA file")
        validate_sequences(seqs)
        return ids, seqs
    finally:
        if temp_fasta_path and Path(temp_fasta_path).exists():
            try:
                os.remove(temp_fasta_path)
            except:
                pass

# -------------------------
# K-mer vectorizer
# -------------------------
class KmerVectorizer:
    def __init__(self, k=4):
        if k < 1 or k > 10:
            raise ValueError("K-mer size must be between 1 and 10")
        self.k = int(k)
        self.vocab = [''.join(p) for p in itertools.product('ATGC', repeat=self.k)]
        self.vocab_size = len(self.vocab)
        print(f"Initialized K-mer vectorizer with k={self.k}, vocab_size={self.vocab_size}")

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
        if not sequences:
            raise ValueError("No sequences provided for transformation")
        X = np.zeros((len(sequences), self.vocab_size), dtype=np.float32)
        for i, seq in enumerate(tqdm(sequences, desc="Vectorizing k-mers")):
            if not seq:
                print(f"Warning: Empty sequence at index {i}")
                continue
            X[i,:] = self.transform_sequence(seq)
        print(f"Vectorization complete: {X.shape[0]} sequences, {X.shape[1]} features")
        return X

# -------------------------
# Dimensionality reduction
# -------------------------
def make_reducer(X, n_components=50):
    if X.shape[0] <= 1:
        n_components = 1
    else:
        n_components = min(n_components, X.shape[0]-1, X.shape[1]-1)
        n_components = max(1, n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xr = svd.fit_transform(X)
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"Dimensionality reduction -> {n_components} components (explained {explained_var:.3f})")
    return svd, Xr

# -------------------------
# Clustering
# -------------------------
def cluster_dbscan(Xr, eps=0.5, min_samples=5):
    print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(Xr)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN: {n_clusters} clusters")
    return labels

def cluster_kmeans(Xr, n_clusters=10):
    max_clusters = min(n_clusters, Xr.shape[0])
    if max_clusters != n_clusters:
        print(f"Adjusted K-means clusters from {n_clusters} to {max_clusters}")
        n_clusters = max_clusters
    print(f"Running KMeans with {n_clusters} clusters")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xr)
    print(f"KMeans results: {len(set(labels))} clusters")
    return labels

# -------------------------
# Medoid selection
# -------------------------
def cluster_medoid_indices(X, labels):
    medoids = {}
    unique_labels = [l for l in set(labels) if l != -1]
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) == 1:
            medoids[lab] = idxs[0]
            continue
        cluster_points = X[idxs]
        centroid = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        med_idx = idxs[np.argmin(dists)]
        medoids[lab] = med_idx
    print(f"Found {len(medoids)} medoids")
    return medoids

# -------------------------
# MLP training
# -------------------------
def train_mlp_sklearn(X, y, epochs=50):
    n_samples = max(1, X.shape[0])
    suggested_batch = max(1, min(64, n_samples // 2))
    print(f"Training MLP: samples={X.shape[0]}, features={X.shape[1]}, batch_size={suggested_batch}, epochs={epochs}")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=suggested_batch,
        learning_rate_init=1e-3,
        max_iter=epochs,
        verbose=False,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    mlp.fit(X, y)
    print(f"MLP training completed. Final loss: {getattr(mlp, 'loss_', 'N/A')}")
    return mlp

# -------------------------
# Training entrypoint
# -------------------------
def train_from_labeled_fasta(train_fasta, outdir, k=4, pca_comp=50, epochs=50, centroid_percentile=95.0):
    print(f"Train: {train_fasta} -> {outdir}")
    outdir_path = Path(outdir); outdir_path.mkdir(parents=True, exist_ok=True)

    ids, seqs = read_fasta_to_list(train_fasta)
    print(f"Loaded {len(seqs)} sequences")

    labels = []
    for header in ids:
        lab = header.split()[0] if header.split() else header
        lab = re.sub(r'[^A-Za-z0-9_]', '_', lab)
        labels.append(lab)
    print(f"Unique labels: {len(set(labels))}")

    vec = KmerVectorizer(k=k)
    X = vec.transform(seqs)

    feature_var = np.var(X, axis=0)
    zero_var_features = np.sum(feature_var == 0)
    if zero_var_features > 0:
        print(f"Warning: {zero_var_features} zero-variance features")

    reducer, Xr = make_reducer(X, n_components=min(pca_comp, X.shape[1]-1 if X.shape[1]>1 else 1, X.shape[0]-1 if X.shape[0]>1 else 1))

    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    class_counts = Counter(y)
    min_samples = 2
    insufficient = [lab for lab, cnt in class_counts.items() if cnt < min_samples]
    if insufficient:
        names = [le.classes_[i] for i in insufficient]
        print(f"Removing classes with <{min_samples} samples: {names}")
        mask = np.isin(y, [lab for lab in class_counts if class_counts[lab] >= min_samples])
        Xr_s = Xr_s[mask]
        kept_labels = [labels[i] for i in range(len(labels)) if mask[i]]
        le = LabelEncoder().fit(kept_labels)
        y = le.transform(kept_labels)

    final_class_count = len(np.unique(y))
    if final_class_count < 2:
        raise ValueError(f"Need at least 2 classes with >= {min_samples} samples; got {final_class_count}")

    print(f"Final dataset: {len(y)} samples, {final_class_count} classes")

    # train/test split
    class_counts = Counter(y)
    min_class_size = min(class_counts.values())
    n_classes = len(class_counts)
    if min_class_size >= 2:
        min_test = max(n_classes, int(0.1 * len(y)))
        max_test = int(0.3 * len(y))
        test_size = min(max_test, max(min_test, int(0.2 * len(y))))
        stratify_param = y
    else:
        test_size = max(1, int(0.2 * len(y)))
        stratify_param = None

    X_train, X_test, y_train, y_test = train_test_split(Xr_s, y, test_size=test_size, random_state=42, stratify=stratify_param)
    print(f"Split: {len(X_train)} train, {len(X_test)} test")

    mlp = train_mlp_sklearn(X_train, y_train, epochs=epochs)

    if len(X_test) > 0:
        preds = mlp.predict(X_test)
        try:
            unique_test_labels = np.unique(y_test)
            target_names = le.inverse_transform(unique_test_labels)
            print(classification_report(y_test, preds, zero_division=0, target_names=target_names))
        except Exception:
            acc = np.mean(preds == y_test)
            print(f"Test accuracy: {acc:.3f}")

    artifacts = {
        "kmer_vectorizer.joblib": vec,
        "reducer.joblib": reducer,
        "scaler.joblib": scaler,
        "label_encoder.joblib": le,
        "mlp_model.joblib": mlp
    }
    for fname, obj in artifacts.items():
        p = outdir_path / fname
        joblib.dump(obj, p)
        print(f"Saved artifact: {p}")

    centroids = {}
    thresholds = {}
    for class_idx, class_label in enumerate(le.classes_):
        class_mask = (y == class_idx)
        if class_mask.sum() == 0:
            continue
        class_points = Xr_s[class_mask]
        centroid = class_points.mean(axis=0)
        distances = np.linalg.norm(class_points - centroid, axis=1)
        centroids[class_label] = centroid
        thresholds[class_label] = float(np.percentile(distances, centroid_percentile))

    joblib.dump({'centroids': centroids, 'thresholds': thresholds}, outdir_path / 'class_centroids.joblib')
    print("Saved centroids and thresholds")

    return str(outdir_path / "mlp_model.joblib")

# -------------------------
# Prediction entrypoint
# -------------------------
def cluster_and_predict(raw_fasta, model_dir, outdir, k=4, pca_comp=50, 
                        cluster_method="kmeans", dbscan_eps=0.5, dbscan_min=5, 
                        kmeans_n=10, threshold=0.7):
    print(f"Predict: {raw_fasta} using model {model_dir} -> {outdir}")
    outdir_path = Path(outdir); outdir_path.mkdir(parents=True, exist_ok=True)
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    ids, seqs = read_fasta_to_list(raw_fasta)
    print(f"Loaded {len(seqs)} sequences for prediction")

    try:
        vec = joblib.load(model_dir_path / "kmer_vectorizer.joblib")
        reducer = joblib.load(model_dir_path / "reducer.joblib")
        scaler = joblib.load(model_dir_path / "scaler.joblib")
        le = joblib.load(model_dir_path / "label_encoder.joblib")
        mlp = joblib.load(model_dir_path / "mlp_model.joblib")
        print("Loaded model artifacts")
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")

    X = vec.transform(seqs)
    Xr = reducer.transform(X)
    Xr_s = scaler.transform(Xr)

    if cluster_method.lower() == "dbscan":
        labels = cluster_dbscan(Xr_s, eps=dbscan_eps, min_samples=dbscan_min)
    else:
        n_clusters = min(kmeans_n, len(Xr_s))
        n_clusters = max(1, n_clusters)
        labels = cluster_kmeans(Xr_s, n_clusters=n_clusters)

    labels = np.array(labels)
    medoids = cluster_medoid_indices(Xr_s, labels)

    centroids_data = None
    centroids_path = model_dir_path / 'class_centroids.joblib'
    if centroids_path.exists():
        centroids_data = joblib.load(centroids_path)
        print("Loaded centroids for validation")

    medoid_rows = []
    for cluster_label, medoid_idx in medoids.items():
        medoid_seq = seqs[medoid_idx]
        medoid_id = ids[medoid_idx]

        x_medoid = X[medoid_idx].reshape(1, -1)
        xr_medoid = reducer.transform(x_medoid)
        xr_s_medoid = scaler.transform(xr_medoid)

        probs = mlp.predict_proba(xr_s_medoid)[0]
        top_indices = probs.argsort()[::-1][:3]

        # Convert top_probs to readable label + probability tuples
        top_probs = []
        top_labels = []
        for i in top_indices:
            class_numeric = mlp.classes_[i]
            try:
                label_name = le.inverse_transform([int(class_numeric)])[0]
            except Exception:
                label_name = str(class_numeric)
            prob_val = float(probs[i])
            top_probs.append((label_name, prob_val))
            top_labels.append(label_name)

        max_prob = float(np.max(probs))
        pred_class_numeric = mlp.classes_[int(np.argmax(probs))]
        try:
            pred_species_raw = le.inverse_transform([pred_class_numeric])[0]
        except Exception:
            pred_species_raw = str(pred_class_numeric)
        pred_species = pred_species_raw if max_prob >= threshold else "Unknown"

        # distance-based validation
        if pred_species != "Unknown" and centroids_data is not None:
            centroids = centroids_data.get('centroids', {})
            thresholds = centroids_data.get('thresholds', {})
            if pred_species in centroids and pred_species in thresholds:
                centroid = centroids[pred_species]
                distance = float(np.linalg.norm(xr_s_medoid.ravel() - centroid))
                if distance > thresholds[pred_species]:
                    print(f"Medoid {medoid_idx} rejected by distance threshold (dist={distance:.3f} > thr={thresholds[pred_species]:.3f})")
                    pred_species = "Unknown"
                    max_prob = 0.0

        # convert top_probs to readable strings for CSV-friendly output
        top_probs_str = "; ".join([f"{lbl} ({p:.3f})" for lbl, p in top_probs])
        top_labels_str = ", ".join(top_labels)

        medoid_rows.append((
            int(cluster_label), int(medoid_idx), medoid_id, pred_species, float(max_prob), top_probs_str, top_labels_str
        ))

    medoid_df = pd.DataFrame(medoid_rows, columns=[
        "cluster", "medoid_index", "medoid_id", "predicted_species",
        "confidence", "top_probs", "top_labels"
    ])
    medoid_file = outdir_path / "cluster_medoid_predictions.csv"
    medoid_df.to_csv(medoid_file, index=False)
    print(f"Saved medoid predictions -> {medoid_file}")

    # sequence-level assignments
    seq_rows = []
    for i, cluster_label in enumerate(labels):
        cluster_pred = medoid_df.loc[medoid_df['cluster'] == cluster_label]
        if len(cluster_pred) == 0:
            assigned = "Unknown"; conf = 0.0
        else:
            assigned = cluster_pred['predicted_species'].iloc[0]
            conf = float(cluster_pred['confidence'].iloc[0])
        seq_rows.append((ids[i], int(cluster_label), assigned, conf))

    assign_df = pd.DataFrame(seq_rows, columns=["sequence_id", "cluster", "predicted_species", "confidence"])
    assign_file = outdir_path / "sequence_cluster_assignments.csv"
    assign_df.to_csv(assign_file, index=False)
    print(f"Saved sequence assignments -> {assign_file}")

    # abundance and visualizations
    try:
        generate_abundance_reports(assign_df, outdir_path)
        generate_visualizations(assign_df, Xr, labels, outdir_path)
    except Exception as e:
        print(f"Warning: visualization or reports generation failed: {e}")

    # print final listing
    try:
        print("Files written to:", [p.name for p in outdir_path.iterdir() if p.is_file()])
    except Exception:
        pass

    return assign_df, medoid_df, labels

# -------------------------
# Abundance reports
# -------------------------
def generate_abundance_reports(assign_df, outdir_path):
    total_reads = len(assign_df)
    species_counts = assign_df.groupby("predicted_species").size().reset_index(name="n_reads")
    species_counts["percentage"] = 100.0 * species_counts["n_reads"] / max(total_reads, 1)
    species_counts = species_counts.sort_values("n_reads", ascending=False).reset_index(drop=True)
    out = outdir_path / "species_abundance_by_reads.csv"
    species_counts.to_csv(out, index=False)
    print(f"Saved abundance by reads -> {out}")

    cluster_sizes = assign_df.groupby("cluster").size().reset_index(name="cluster_size")
    cluster_info = assign_df.merge(cluster_sizes, on="cluster").drop_duplicates(subset=["cluster"])
    cluster_counts = cluster_info.groupby("predicted_species")["cluster_size"].sum().reset_index(name="n_reads_via_clusters")
    cluster_counts["percentage_via_clusters"] = 100.0 * cluster_counts["n_reads_via_clusters"] / max(total_reads, 1)
    cluster_counts = cluster_counts.sort_values("n_reads_via_clusters", ascending=False)
    out2 = outdir_path / "species_abundance_by_clusters.csv"
    cluster_counts.to_csv(out2, index=False)
    print(f"Saved abundance by clusters -> {out2}")

# -------------------------
# Visualizations
# -------------------------
def generate_visualizations(assign_df, Xr, labels, outdir_path):
    try:
        species_counts = assign_df.groupby("predicted_species").size().reset_index(name="n_reads")
        total_reads = len(assign_df)
        species_counts["percentage"] = 100.0 * species_counts["n_reads"] / max(total_reads, 1)
        species_counts = species_counts.sort_values("n_reads", ascending=False).reset_index(drop=True)

        # bar
        if len(species_counts) > 0:
            plt.figure(figsize=(10, max(6, 0.4 * min(20, len(species_counts)))))
            top_n = min(20, len(species_counts))
            ax = sns.barplot(x="percentage", y="predicted_species", data=species_counts.head(top_n), orient="h")
            plt.xlabel("Percentage of reads (%)"); plt.ylabel("Predicted species")
            plt.title(f"Species abundance (top {top_n}) by reads")
            for p in ax.patches:
                width = p.get_width()
                if width > 0:
                    ax.text(width + 0.5, p.get_y() + p.get_height()/2, f"{width:.1f}%", va='center')
            plt.tight_layout()
            barpath = outdir_path / "species_abundance_bar.png"
            plt.savefig(barpath, dpi=300, bbox_inches='tight'); plt.close()
            print(f"Saved bar chart -> {barpath}")

        # pie
        if len(species_counts) > 0:
            pie_df = species_counts.copy()
            top_n = min(20, len(pie_df))
            if len(pie_df) > top_n:
                top_df = pie_df.head(top_n).copy()
                others_pct = pie_df['percentage'].iloc[top_n:].sum()
                if others_pct > 0:
                    top_df = pd.concat([top_df, pd.DataFrame([{'predicted_species': 'Others', 'n_reads': 0, 'percentage': others_pct}])], ignore_index=True)
            else:
                top_df = pie_df
            if len(top_df) > 0 and top_df['percentage'].sum() > 0:
                fig, ax = plt.subplots(figsize=(8,8))
                labels_pie = top_df['predicted_species'].tolist()
                sizes = top_df['percentage'].tolist()
                labels_display = [label if size >= 2 else '' for label, size in zip(labels_pie, sizes)]
                wedges, texts, autotexts = ax.pie(sizes, labels=labels_display,
                                                 autopct=lambda pct: f'{pct:.1f}%' if pct >= 1 else '',
                                                 startangle=90, pctdistance=0.85)
                ax.axis('equal')
                plt.title('Species composition (by reads)')
                piepath = outdir_path / 'species_composition_pie.png'
                plt.savefig(piepath, dpi=300, bbox_inches='tight'); plt.close()
                print(f"Saved pie chart -> {piepath}")

        # scatter using first two components
        if Xr.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10,8))
            unique_labels = sorted(set(labels))
            palette = sns.color_palette("tab10", n_colors=len(unique_labels)) if len(unique_labels) <= 10 else sns.color_palette("hls", n_colors=len(unique_labels))
            for idx, lab in enumerate(unique_labels):
                mask = labels == lab
                color = palette[idx % len(palette)]
                label_name = f"Cluster {lab}" if lab != -1 else "Noise"
                ax.scatter(Xr[mask, 0], Xr[mask, 1], s=20, color=color, label=label_name, alpha=0.7)
            if len(unique_labels) <= 15:
                ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize='small')
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Clustering scatter (PC1 vs PC2)")
            plt.tight_layout()
            scatterpath = outdir_path / "cluster_scatter.png"
            plt.savefig(scatterpath, dpi=300, bbox_inches='tight'); plt.close()
            print(f"Saved scatter plot -> {scatterpath}")

    except Exception as e:
        print(f"Error generating visuals: {e}")
        import traceback; traceback.print_exc()

# -------------------------
# Utility
# -------------------------
def get_model_info(model_dir):
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        return None
    info = {"model_dir": str(model_dir_path), "files_present": {}}
    required_files = ["kmer_vectorizer.joblib", "reducer.joblib", "scaler.joblib", "label_encoder.joblib", "mlp_model.joblib"]
    for fname in required_files:
        info["files_present"][fname] = (model_dir_path / fname).exists()
    try:
        if info["files_present"]["kmer_vectorizer.joblib"]:
            vec = joblib.load(model_dir_path / "kmer_vectorizer.joblib")
            info["k_mer_size"] = vec.k; info["vocabulary_size"] = vec.vocab_size
        if info["files_present"]["label_encoder.joblib"]:
            le = joblib.load(model_dir_path / "label_encoder.joblib")
            info["num_classes"] = len(le.classes_); info["class_names"] = le.classes_.tolist()
        if info["files_present"]["reducer.joblib"]:
            reducer = joblib.load(model_dir_path / "reducer.joblib"); info["pca_components"] = getattr(reducer, "n_components", None)
    except Exception as e:
        info["parameter_loading_error"] = str(e)
    return info

if __name__ == "__main__":
    print("Pipeline module ready.")
