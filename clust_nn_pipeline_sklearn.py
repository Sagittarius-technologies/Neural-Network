#!/usr/bin/env python3
"""
clust_nn_pipeline_sklearn.py

End-to-end pipeline using scikit-learn MLP (no TensorFlow required):
- k-mer vectorization
- TruncatedSVD dimensionality reduction
- DBSCAN or KMeans clustering
- Medoid selection per cluster
- MLPClassifier training on labeled FASTA
- Classify medoids, propagate labels, compute abundance & plots

Usage examples:

Train:
python clust_nn_pipeline_sklearn.py --train_fasta sequences_clean.fasta --outdir model_output --k 4 --pca 50 --epochs 50

Cluster & predict:
python clust_nn_pipeline_sklearn.py --raw_fasta raw_sequences.fasta --outdir results --k 4 --pca 50 --cluster_method dbscan --dbscan_eps 0.5 --dbscan_min 3 --threshold 0.7

Train + cluster in one go:
python clust_nn_pipeline_sklearn.py --train_fasta sequences_clean.fasta --raw_fasta raw_sequences.fasta --outdir results --k 4 --pca 50 --epochs 50 --cluster_method dbscan --dbscan_eps 0.5 --dbscan_min 3 --threshold 0.7
"""
import os, sys, argparse, glob, re
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
    km = KMeans(n_clusters=n_clusters, random_state=42)
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
def train_from_labeled_fasta(train_fasta, outdir, k=4, pca_comp=50, epochs=30, centroid_percentile=95.0):
    os.makedirs(outdir, exist_ok=True)
    fasta = ensure_fasta_headers(train_fasta)
    ids, seqs = read_fasta_to_list(fasta)
    labels = []
    for header in ids:
        parts = header.split()
        lab = parts[0] if len(parts)>=1 else header
        lab = re.sub(r'[^A-Za-z0-9_]', '_', lab)
        labels.append(lab)
    vec = KmerVectorizer(k=k)
    X = vec.transform(seqs)
    reducer, Xr = make_reducer(X, n_components=min(pca_comp, X.shape[1]-1))
    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    counts = Counter(y)
    bad = [lab for lab,c in counts.items() if c < 2]
    if bad:
        print("Warning: following classes have <2 samples and will be removed:", bad)
        keep_mask = np.isin(y, [lab for lab in counts if counts[lab]>=2])
        Xr_s = Xr_s[keep_mask]
        y = y[keep_mask]
        labels_kept = [labels[i] for i in range(len(labels)) if keep_mask[i]]
        le = LabelEncoder().fit(labels_kept)
        y = le.transform(labels_kept)
    # split and train
    X_train, X_test, y_train, y_test = train_test_split(Xr_s, y, test_size=0.2, random_state=42, stratify=y)
    mlp = train_mlp_sklearn(X_train, y_train, epochs=epochs)
    preds = mlp.predict(X_test)
    print("Classification report on test set:")
    print(classification_report(y_test, preds, zero_division=0, target_names=le.inverse_transform(np.unique(y_test))))
    # save artifacts
    joblib.dump(vec, os.path.join(outdir, "kmer_vectorizer.joblib"))
    joblib.dump(reducer, os.path.join(outdir, "reducer.joblib"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.joblib"))
    joblib.dump(le, os.path.join(outdir, "label_encoder.joblib"))
    joblib.dump(mlp, os.path.join(outdir, "mlp_model.joblib"))
    # compute per-class centroids and save distance thresholds (95th percentile)
    centroids = {}
    thresholds = {}
    for class_idx, class_label in enumerate(le.classes_):
        mask = (y == class_idx)
        if mask.sum() == 0:
            continue
        Xc = Xr_s[mask]
        centroid = Xc.mean(axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        centroids[class_label] = centroid
        thresholds[class_label] = float(np.percentile(dists, float(centroid_percentile)))
    joblib.dump({'centroids': centroids, 'thresholds': thresholds}, os.path.join(outdir, 'class_centroids.joblib'))
    print("Saved artifacts to", outdir)
    return os.path.join(outdir, "mlp_model.joblib")

def cluster_and_predict(raw_fasta, model_path, vec_path, reducer_path, scaler_path, encoder_path,
                        outdir, k=4, cluster_method="dbscan", dbscan_eps=0.5, dbscan_min=5,
                        kmeans_n=10, pca_comp=50, threshold=0.7):
    os.makedirs(outdir, exist_ok=True)
    fasta = ensure_fasta_headers(raw_fasta, tmp_file=os.path.join(outdir, "tmp_with_headers.fasta"))
    ids, seqs = read_fasta_to_list(fasta)
    vec = joblib.load(vec_path)
    X = vec.transform(seqs)
    reducer = joblib.load(reducer_path)
    Xr = reducer.transform(X)
    scaler = joblib.load(scaler_path)
    Xr_s = scaler.transform(Xr)
    if cluster_method.lower() == "dbscan":
        labels = cluster_dbscan(Xr_s, eps=dbscan_eps, min_samples=dbscan_min)
    else:
        labels = cluster_kmeans(Xr_s, n_clusters=kmeans_n)
    labels = np.array(labels)
    print("Clusters found:", len(set(labels)) - (1 if -1 in labels else 0), "plus noise (-1):", sum(labels==-1))
    medoids = cluster_medoid_indices(Xr_s, labels)
    mlp = joblib.load(model_path)
    le = joblib.load(encoder_path)
    # load centroids and thresholds if available
    centroids_path = os.path.join(outdir, 'class_centroids.joblib')
    centroids_data = None
    if os.path.exists(centroids_path):
        centroids_data = joblib.load(centroids_path)
    medoid_rows = []
    for cl, midx in medoids.items():
        seq = seqs[midx]
        seq_id = ids[midx]
        x = vec.transform([seq])
        xr = reducer.transform(x)
        xr_s = scaler.transform(xr)
        probs = mlp.predict_proba(xr_s)[0]
        # top-3 for reporting
        top_idx = probs.argsort()[::-1][:3]
        top_probs = [(mlp.classes_[i], float(probs[i])) for i in top_idx]
        top_labels = [le.inverse_transform([int(c)])[0] for c,_ in top_probs]
        maxp = float(np.max(probs))
        pred_numeric = mlp.classes_[int(np.argmax(probs))]
        pred_label = le.inverse_transform([pred_numeric])[0] if maxp >= threshold else "Unknown"
        # centroid distance reject: if centroids available, check medoid distance
        if pred_label != "Unknown" and centroids_data is not None:
            centroids = centroids_data.get('centroids', {})
            thresholds = centroids_data.get('thresholds', {})
            if pred_label in centroids:
                centroid = centroids[pred_label]
                dist = float(np.linalg.norm(xr_s.ravel() - centroid))
                # if distance greater than stored threshold, mark Unknown
                if pred_label in thresholds and dist > thresholds[pred_label]:
                    pred_label = "Unknown"
                    maxp = 0.0
        medoid_rows.append((cl, midx, seq_id, pred_label, maxp, top_probs, top_labels))
    medoid_df = pd.DataFrame(medoid_rows, columns=["cluster","medoid_index","medoid_id","predicted_species","confidence","top_probs","top_labels"])
    medoid_df.to_csv(os.path.join(outdir, "cluster_medoid_predictions.csv"), index=False)
    # propagate
    cluster_assignments = []
    for i, lab in enumerate(labels):
        assigned = medoid_df.loc[medoid_df['cluster'] == lab, 'predicted_species'].values
        assigned_conf = medoid_df.loc[medoid_df['cluster'] == lab, 'confidence'].values
        if len(assigned) == 0:
            assigned_species = "Unknown"
            conf = 0.0
        else:
            assigned_species = assigned[0]
            conf = assigned_conf[0]
        cluster_assignments.append((ids[i], lab, assigned_species, conf))
    assign_df = pd.DataFrame(cluster_assignments, columns=["sequence_id","cluster","predicted_species","confidence"])
    assign_df.to_csv(os.path.join(outdir, "sequence_cluster_assignments.csv"), index=False)
    print("Saved sequence assignments.")
    # abundance + plots (inserted)
    try:
        total_reads = len(assign_df)
        species_counts = assign_df.groupby("predicted_species").size().reset_index(name="n_reads")
        species_counts["percentage"] = 100.0 * species_counts["n_reads"] / total_reads
        species_counts = species_counts.sort_values("n_reads", ascending=False).reset_index(drop=True)
        species_counts.to_csv(os.path.join(outdir, "species_abundance_by_reads.csv"), index=False)
        print("Saved species_abundance_by_reads.csv")
        cluster_sizes = assign_df.groupby("cluster").size().reset_index(name="cluster_size")
        cluster_info = assign_df.merge(cluster_sizes, on="cluster").drop_duplicates(subset=["cluster"])
        cluster_counts = cluster_info.groupby("predicted_species")["cluster_size"].sum().reset_index(name="n_reads_via_clusters")
        cluster_counts["percentage_via_clusters"] = 100.0 * cluster_counts["n_reads_via_clusters"] / total_reads
        cluster_counts = cluster_counts.sort_values("n_reads_via_clusters", ascending=False)
        cluster_counts.to_csv(os.path.join(outdir, "species_abundance_by_clusters.csv"), index=False)
        # bar plot with percentage annotations (sorted descending)
        plt.figure(figsize=(8, max(4, 0.4 * min(20, len(species_counts)))))
        topN = min(20, len(species_counts))
        ax = sns.barplot(x="percentage", y="predicted_species", data=species_counts.head(topN), orient="h")
        plt.xlabel("Percentage of reads (%)")
        plt.ylabel("Predicted species")
        plt.title(f"Species abundance (top {topN}) by reads")
        # annotate bars with percentages
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.5, p.get_y() + p.get_height() / 2, f"{width:.1f}%", va='center')
        plt.tight_layout()
        barpath = os.path.join(outdir, "species_abundance_bar.png")
        plt.savefig(barpath)
        plt.close()
        print("Saved species abundance bar chart.")

        # pie chart showing composition (topN, aggregate others if present)
        try:
            pie_df = species_counts.copy()
            if len(pie_df) > topN:
                top_df = pie_df.head(topN)
                others_pct = pie_df['percentage'].iloc[topN:].sum()
                top_df = top_df.append({'predicted_species': 'Others', 'n_reads': 0, 'percentage': others_pct}, ignore_index=True)
            else:
                top_df = pie_df
            labels = top_df['predicted_species'].tolist()
            sizes = top_df['percentage'].tolist()
            fig, ax = plt.subplots(figsize=(6,6))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.8)
            ax.axis('equal')
            plt.title('Species composition (by reads)')
            piepath = os.path.join(outdir, 'species_composition_pie.png')
            plt.savefig(piepath)
            plt.close()
            print('Saved species composition pie chart.')
        except Exception as e:
            print('Could not make pie chart:', e)
    except Exception as e:
        print("Could not compute/save species abundance plots:", e)
    # scatter
    try:
        fig, ax = plt.subplots(figsize=(8,6))
        unique = sorted(set(labels))
        palette = sns.color_palette("hls", n_colors=len(unique))
        for idx, lab in enumerate(unique):
            mask = labels == lab
            ax.scatter(Xr[mask,0], Xr[mask,1], s=10, color=palette[idx], label=str(lab))
        ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize='small')
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Clustering scatter (first 2 reduced components)")
        plt.tight_layout()
        scatter_path = os.path.join(outdir, "cluster_scatter.png")
        plt.savefig(scatter_path)
        plt.close()
        print("Saved cluster scatter plot.")
    except Exception as e:
        print("Could not make scatter plot:", e)
    return assign_df, medoid_df, labels

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Unsupervised clustering + sklearn-MLP classification pipeline for eDNA")
    parser.add_argument("--raw_fasta", help="Raw mixed sequences (FASTA or plain lines)", default=None)
    parser.add_argument("--train_fasta", help="Labeled reference FASTA (for training MLP)", default=None)
    parser.add_argument("--outdir", help="Output directory", default="results")
    parser.add_argument("--k", help="k for k-mer (default 4)", type=int, default=4)
    parser.add_argument("--pca", help="PCA components (TruncatedSVD) default 50", type=int, default=50)
    parser.add_argument("--cluster_method", help="dbscan or kmeans", default="dbscan")
    parser.add_argument("--dbscan_eps", help="DBSCAN eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min", help="DBSCAN min_samples", type=int, default=5)
    parser.add_argument("--kmeans_n", help="KMeans n_clusters (if chosen)", type=int, default=10)
    parser.add_argument("--threshold", help="Confidence threshold for NN predictions", type=float, default=0.7)
    parser.add_argument("--epochs", help="MLP training epochs (max_iter)", type=int, default=50)
    parser.add_argument("--centroid_percentile", help="Percentile for centroid distance threshold (default 95)", type=float, default=95.0)
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    model_path = None
    if args.train_fasta:
        print("Training MLP from labeled FASTA...")
        model_path = train_from_labeled_fasta(args.train_fasta, outdir, k=args.k, pca_comp=args.pca, epochs=args.epochs, centroid_percentile=args.centroid_percentile)

    if args.raw_fasta:
        vec_path = os.path.join(outdir, "kmer_vectorizer.joblib")
        reducer_path = os.path.join(outdir, "reducer.joblib")
        scaler_path = os.path.join(outdir, "scaler.joblib")
        encoder_path = os.path.join(outdir, "label_encoder.joblib")
        if not model_path:
            if not os.path.exists(vec_path) or not os.path.exists(reducer_path) or not os.path.exists(scaler_path) or not os.path.exists(encoder_path) or not os.path.exists(os.path.join(outdir, "mlp_model.joblib")):
                print("ERROR: artifacts not found in outdir. Either provide --train_fasta or ensure artifacts exist in outdir.")
                sys.exit(1)
            model_path = os.path.join(outdir, "mlp_model.joblib")
        assign_df, medoid_df, labels = cluster_and_predict(
            args.raw_fasta,
            model_path,
            vec_path,
            reducer_path,
            scaler_path,
            encoder_path,
            outdir,
            k=args.k,
            cluster_method=args.cluster_method,
            dbscan_eps=args.dbscan_eps,
            dbscan_min=args.dbscan_min,
            kmeans_n=args.kmeans_n,
            pca_comp=args.pca,
            threshold=args.threshold
        )

if __name__ == "__main__":
    main()
