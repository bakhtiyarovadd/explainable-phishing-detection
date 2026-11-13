"""Leave-one-source-out test: train on all datasets except one, test on the one that was left out.

This is the honest way to combine several datasets: every dataset is tested on emails whose *source* the model
has never seen. Each CSV needs the columns sender, subject, body, label (1 = phishing/spam, 0 = legitimate).

Usage:
  python scripts/leave_one_source_out.py --data data/CEAS_08.csv data/Nazario.csv data/Enron.csv data/SpamAssasin.csv
Options: --max-per-source 4000 (random sample per file, keeps it fast)
Single-class files (e.g. only phishing) are reported with recall only (or only 'legit kept' if all legitimate).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phishing_xai.features import clean_text  # noqa: E402


def load(path, n, seed):
    df = pd.read_csv(path)
    for c in ("sender", "subject"):
        if c not in df:
            df[c] = ""
    df = df[df["label"].isin([0, 1])]
    df = df[df["body"].fillna("").str.strip().str.len() > 0].drop_duplicates(subset=["subject", "body"])
    if len(df) > n:
        df = df.sample(n, random_state=seed)
    df = df.assign(text=[clean_text(s, b) for s, b in zip(df["subject"], df["body"])], label=df["label"].astype(int))
    return df[["text", "label"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--max-per-source", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sets = {Path(p).stem: load(p, a.max_per_source, a.seed) for p in a.data}
    for k, v in sets.items():
        print(f"{k:16s} {len(v):6,} emails  ({int(v.label.sum()):,} phishing/spam, {int((v.label == 0).sum()):,} legitimate)")
    print("\n| Held-out source | Model | Accuracy | Phishing/spam caught | Legitimate kept |\n|---|---|---|---|---|")
    for held in sets:
        train = pd.concat([v for k, v in sets.items() if k != held], ignore_index=True)
        test = sets[held]
        if train.label.nunique() < 2:
            print(f"| {held} | (skipped: the other sources have only one class) | | | |")
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=30000, sublinear_tf=True, stop_words="english")
        Xtr, Xte = vec.fit_transform(train["text"]), vec.transform(test["text"])
        for name, model in (("Logistic Regression", LogisticRegression(C=5, max_iter=2000, solver="liblinear")),
                            ("Linear SVM", LinearSVC(C=0.5))):
            pred = model.fit(Xtr, train["label"]).predict(Xte)
            y = test["label"].values
            acc = (pred == y).mean() * 100
            rec = f"{(pred[y == 1] == 1).mean() * 100:.1f}%" if (y == 1).any() else "\u2013"
            spec = f"{(pred[y == 0] == 0).mean() * 100:.1f}%" if (y == 0).any() else "\u2013"
            print(f"| {held} | {name} | {acc:.1f}% | {rec} | {spec} |")


if __name__ == "__main__":
    main()
