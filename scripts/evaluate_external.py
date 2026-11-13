"""Test the already-trained models on ANOTHER dataset (cross-dataset test, no retraining).

Usage:  python scripts/evaluate_external.py --data data/other.csv   [--tag name]
The CSV needs the columns sender, subject, body, label (1 = phishing/spam, 0 = legitimate).
If your column names differ, use --sender-col / --subject-col / --body-col / --label-col.
To combine a phishing CSV and a legitimate CSV first:  python scripts/evaluate_external.py --data a.csv b.csv
Output: results/external_results.json and a Markdown table in the terminal.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phishing_xai.explain import email_inputs, predict_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--sender-col", default="sender")
    ap.add_argument("--subject-col", default="subject")
    ap.add_argument("--body-col", default="body")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-rows", type=int, default=5000, help="random sample size (keeps it fast)")
    ap.add_argument("--tag", default="", help="use models/bundle_<tag>.joblib made by: train.py --tag <tag>")
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""

    df = pd.concat([pd.read_csv(p) for p in a.data], ignore_index=True)
    df = df.rename(columns={a.sender_col: "sender", a.subject_col: "subject", a.body_col: "body", a.label_col: "label"})
    for c in ("sender", "subject"):
        if c not in df:
            df[c] = ""
    df = df[df["body"].fillna("").str.strip().str.len() > 0].drop_duplicates(subset=["subject", "body"])
    if len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=0)
    y = df["label"].astype(int).values
    print(f"{len(df):,} emails ({y.sum():,} phishing/spam, {(y == 0).sum():,} legitimate)")

    bundle = joblib.load(ROOT / "models" / f"bundle{suf}.joblib")
    probs = {m: [] for m in bundle["models"]}
    for s, sub, b in zip(df["sender"], df["subject"], df["body"]):
        t, f, fr = email_inputs(bundle, s, sub, b)
        for m, p in predict_all(bundle, t, fr).items():
            probs[m].append(p)
    out = {"n": int(len(df)), "n_phishing": int(y.sum()), "results": {}}
    print("\n| Model | Accuracy | Recall (phishing caught) | FPR (legit flagged) |\n|---|---|---|---|")
    for m, p in probs.items():
        pred = (np.array(p) >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        r = {"accuracy": (tp + tn) / len(y), "recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1)}
        out["results"][m] = r
        print(f"| {m} | {r['accuracy'] * 100:.1f}% | {r['recall'] * 100:.1f}% | {r['fpr'] * 100:.1f}% |")
    (ROOT / "results" / f"external_results{suf}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
