"""Check a CSV before using it: size, columns, labels, empty bodies, duplicates.

Usage:  python scripts/check_dataset.py data/Nazario.csv data/Enron.csv
Required columns: subject, body, label (sender is optional). Labels must be 1 (phishing/spam) or 0 (legitimate).
"""
import sys

import pandas as pd

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    print(f"\n== {path}\nrows: {len(df):,}\ncolumns: {list(df.columns)}")
    missing = [c for c in ("subject", "body", "label") if c not in df.columns]
    if missing:
        print("PROBLEM: missing column(s):", missing, "-> rename them first (see the instructions)")
        continue
    print("label values:", df["label"].value_counts(dropna=False).to_dict())
    if not set(df["label"].dropna().unique()) <= {0, 1}:
        print("PROBLEM: labels must be 0 or 1 -> convert them first")
    body = df["body"].fillna("")
    print("empty bodies:", int((body.str.strip().str.len() == 0).sum()))
    print("exact duplicates (subject + body):", int(df.duplicated(subset=["subject", "body"]).sum()))
    print("median body length (characters):", int(body.str.len().median()))
