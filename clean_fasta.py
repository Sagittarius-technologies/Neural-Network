from Bio import SeqIO
import glob, os, re

fasta_dir = "Datasets"
out_file = "sequences_clean.fasta"

def clean_label(desc):
    # Normalize and remove common prefixes like 'PREDICTED:' (various styles)
    desc = re.sub(r'(?i)PREDICTED[:_]*', '', desc)
    # Remove leading accession-like token (e.g. XR_006403649.1 or similar)
    desc = re.sub(r'^[^\s]+\s+', '', desc)
    # Now take genus and optionally species as the label
    parts = desc.split()
    if len(parts) >= 2 and re.match(r'^[A-Za-z-]+$', parts[0]) and re.match(r'^[A-Za-z-]+$', parts[1]):
        label = parts[0] + "_" + parts[1]
    elif len(parts) >= 1 and re.match(r'^[A-Za-z-]+$', parts[0]):
        label = parts[0]
    else:
        # fallback: use cleaned alphanumeric from entire desc
        label = re.sub(r'[^A-Za-z_]', '_', desc).strip('_') or 'unknown'
    # final sanitization: keep letters and underscore only
    label = re.sub(r'[^A-Za-z_]', '', label)
    return label.strip()

with open(out_file, "w") as out_handle:
    for fasta_file in glob.glob(os.path.join(fasta_dir, "*.fasta")):
        for rec in SeqIO.parse(fasta_file, "fasta"):
            label = clean_label(rec.description)
            rec.id = label
            rec.description = label
            SeqIO.write(rec, out_handle, "fasta")

print(f"Cleaned FASTA saved as {out_file}")
