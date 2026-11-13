"""Combine several CSV files into one training file (columns: sender, subject, body, label).

Usage:
  python scripts/combine_datasets.py --files data/Nazario.csv data/Nigerian_Fraud.csv data/Enron.csv --out data/combined.csv
Options: --max-per-file 5000  (random sample per file, keeps classes from being dominated by one big file)
Exact duplicates across files are removed. The script warns if a class comes from only ONE source file,
because the model may then learn the source instead of the threat.
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-per-file", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

parts, rows = [], []
for f in a.files:
    df = pd.read_csv(f)
    for c in ("sender", "subject"):
        if c not in df:
            df[c] = ""
    df = df[df["label"].isin([0, 1])]
    df = df[df["body"].fillna("").str.strip().str.len() > 0]
    if len(df) > a.max_per_file:
        df = df.sample(a.max_per_file, random_state=a.seed)
    df = df[["sender", "subject", "body", "label"]].assign(label=lambda x: x["label"].astype(int), source=Path(f).stem)
    parts.append(df)
    rows.append((Path(f).stem, int((df.label == 1).sum()), int((df.label == 0).sum())))
out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["subject", "body"])
out.drop(columns="source").to_csv(a.out, index=False)
print(f"{'file':18s} {'phishing/spam':>14s} {'legitimate':>11s}")
for n, p, l in rows:
    print(f"{n:18s} {p:14,d} {l:11,d}")
print(f"\nSaved {a.out}: {len(out):,} emails ({int(out.label.sum()):,} phishing/spam, {int((out.label == 0).sum()):,} legitimate)")
for lab, name in ((1, "phishing/spam"), (0, "legitimate")):
    srcs = out[out.label == lab]["source"].nunique()
    if srcs < 2:
        print(f"WARNING: all {name} emails come from ONE source file; the model may learn the source instead of the threat. "
              f"Add another file that contains {name} emails.")
