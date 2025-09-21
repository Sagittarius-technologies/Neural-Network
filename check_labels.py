from Bio import SeqIO
from collections import Counter
import glob
import os

fasta_dir = "sequences_clean.fasta"
labels = []

# Loop through all fasta files in the folder
for fasta_file in glob.glob(os.path.join(fasta_dir, "*.fasta")):
    print(f"Parsing {fasta_file} ...")
    for rec in SeqIO.parse(fasta_file, "fasta"):
        # Adjust parsing based on header style
        words = rec.description.split()
        if len(words) >= 2:
            label = "_".join(words[1:3])  # e.g., Escherichia_coli
        else:
            label = rec.id
        labels.append(label)

# Count species
counts = Counter(labels)
print("\n=== Summary ===")
print("Total sequences:", sum(counts.values()))
print("Unique labels:", len(counts))
print("Label counts (first 50):")
for i, (lab, c) in enumerate(counts.most_common()):
    print(f"{lab}: {c}")
    if i >= 49:
        break
