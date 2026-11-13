"""Train and evaluate the traditional ML models on the CEAS-08 corpus.

Usage:  python scripts/train.py [--data data/CEAS_08.csv] [--seed 42] [--tag name]

Outputs
  models/bundle.joblib        vectoriser, feature selector and all fitted models
  results/metrics.json        test-set metrics, 5-fold CV scores, timings
  results/test_ids.csv        row ids of the held-out test split (used by the LLM evaluation)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phishing_xai.features import FEATURE_NAMES, clean_text, feature_frame  # noqa: E402

N_TFIDF_FOR_TREES = 3000


def build_models(seed: int):
    return {
        "Logistic Regression (TF-IDF)": ("text", lambda: LogisticRegression(C=5, max_iter=2000, solver="liblinear")),
        "Linear SVM (TF-IDF)": ("text", lambda: CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
        "Naive Bayes (TF-IDF)": ("text", lambda: MultinomialNB(alpha=0.1)),
        "Random Forest (handcrafted)": ("feat", lambda: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed)),
        "XGBoost (handcrafted)": ("feat", lambda: XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", n_jobs=1, random_state=seed, eval_metric="logloss")),
        "XGBoost (TF-IDF + handcrafted)": ("hybrid", lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.5,
            tree_method="hist", n_jobs=1, random_state=seed, eval_metric="logloss")),
    }


def view(kind, X_text, X_sel, X_feat):
    if kind == "text":
        return X_text
    if kind == "feat":
        return X_feat
    return sparse.hstack([X_sel, sparse.csr_matrix(X_feat)]).tocsr()


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob),
        "fpr": fp / (fp + tn), "fnr": fn / (fn + tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "CEAS_08.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cv", action="store_true", help="skip 5-fold cross-validation")
    ap.add_argument("--tag", default="", help="name for a separate run, e.g. --tag nazario; outputs get this suffix "
                                              "so the default models and results are NOT overwritten")
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""

    t0 = time.time()
    df = pd.read_csv(args.data)
    df["row_id"] = np.arange(len(df))
    n_read = len(df)
    # ---- record rejection criteria (each rule is counted and printed) ----
    bad_label = ~df["label"].isin([0, 1])
    df = df[~bad_label]
    empty_body = ~(df["body"].notna() & (df["body"].str.strip().str.len() > 0))
    df = df[~empty_body]
    dup = df.duplicated(subset=["subject", "body"])
    df = df[~dup].reset_index(drop=True)
    print(f"Rejected records: missing/invalid label = {int(bad_label.sum())}, empty body = {int(empty_body.sum())}, "
          f"exact duplicate (same subject + body) = {int(dup.sum())}  ->  kept {len(df):,} of {n_read:,}")
    print(f"Loaded {len(df):,} emails  ({df.label.sum():,} phishing/spam, {(df.label == 0).sum():,} legitimate)")

    print("Extracting features ...")
    df["text"] = [clean_text(s, b) for s, b in zip(df["subject"], df["body"])]
    F = feature_frame(df)

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["label"], random_state=args.seed)
    y = df["label"].values
    y_tr, y_te = y[idx_tr], y[idx_te]
    pd.DataFrame({"row_id": df.loc[idx_te, "row_id"].values, "label": y_te}).to_csv(ROOT / "results" / f"test_ids{suf}.csv", index=False)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=30000, sublinear_tf=True,
                            stop_words="english")
    Xt_tr = tfidf.fit_transform(df.loc[idx_tr, "text"])
    Xt_te = tfidf.transform(df.loc[idx_te, "text"])
    sel = SelectKBest(chi2, k=N_TFIDF_FOR_TREES).fit(Xt_tr, y_tr)
    Xs_tr, Xs_te = sel.transform(Xt_tr), sel.transform(Xt_te)
    F_tr, F_te = F.iloc[idx_tr].values, F.iloc[idx_te].values
    print(f"TF-IDF vocabulary: {len(tfidf.vocabulary_):,} terms; train {len(idx_tr):,} / test {len(idx_te):,}")

    results, fitted, roc_curves = {}, {}, {}
    from sklearn.metrics import roc_curve
    for name, (kind, factory) in build_models(args.seed).items():
        t1 = time.time()
        model = factory()
        A_tr = view(kind, Xt_tr, Xs_tr, F_tr)
        A_te = view(kind, Xt_te, Xs_te, F_te)
        model.fit(A_tr, y_tr)
        train_s = time.time() - t1
        t2 = time.time()
        prob = model.predict_proba(A_te)[:, 1]
        infer_ms = (time.time() - t2) / len(y_te) * 1000
        m = metrics(y_te, prob)
        m.update(train_seconds=train_s, infer_ms_per_email=infer_ms, kind=kind)

        if not args.no_cv:
            skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
            f1s = []
            for a, b in skf.split(A_tr, y_tr):
                mm = factory().fit(A_tr[a], y_tr[a])
                f1s.append(f1_score(y_tr[b], mm.predict(A_tr[b])))
            m["cv_f1_mean"], m["cv_f1_std"] = float(np.mean(f1s)), float(np.std(f1s))
        results[name] = m
        fitted[name] = model
        fpr, tpr, _ = roc_curve(y_te, prob)
        step = max(1, len(fpr) // 400)
        roc_curves[name] = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}
        print(f"{name:34s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
              f"fpr={m['fpr']:.4f} fnr={m['fnr']:.4f} train={train_s:.1f}s")

    best = max(results, key=lambda k: results[k]["f1"])
    bundle = {"tfidf": tfidf, "selector": sel, "models": fitted, "kinds": {k: v[0] for k, v in build_models(args.seed).items()},
              "feature_names": FEATURE_NAMES, "best": best, "seed": args.seed}
    joblib.dump(bundle, ROOT / "models" / f"bundle{suf}.joblib", compress=3)

    # background sample for SHAP / feature-importance plots
    pd.DataFrame(F_te, columns=FEATURE_NAMES).assign(label=y_te).to_csv(ROOT / "results" / f"test_features{suf}.csv", index=False)

    out = {"dataset": {"n_emails": int(len(df)), "n_phishing": int(y.sum()), "n_legit": int((y == 0).sum()),
                       "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)), "tfidf_terms": len(tfidf.vocabulary_),
                       "n_handcrafted_features": len(FEATURE_NAMES)},
           "models": results, "best_model": best, "roc": roc_curves, "seed": args.seed,
           "total_seconds": time.time() - t0}
    (ROOT / "results" / f"metrics{suf}.json").write_text(json.dumps(out, indent=2))
    print(f"\nBest model by F1: {best}\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
