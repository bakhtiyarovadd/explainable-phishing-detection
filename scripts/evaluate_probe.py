"""Out-of-distribution probe: evaluate every model on data/probe_set.csv (24 hand-written modern emails).

The probe set is small and synthetic. It is meant to show domain shift (modern credential / BEC / delivery
phishing vs. the 2008 CEAS corpus), NOT to serve as a benchmark.
Usage:  python scripts/evaluate_probe.py            # ML models only
        python scripts/evaluate_probe.py --llm      # also run the LLM (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phishing_xai.explain import email_inputs, predict_all  # noqa: E402
from phishing_xai.llm import DEFAULT_MODEL, analyze_email  # noqa: E402


def summ(y, p):
    pred = (np.asarray(p) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "fpr": fp / max(fp + tn, 1),
            "fnr": fn / max(fn + tp, 1), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    bundle = joblib.load(ROOT / "models" / "bundle.joblib")
    df = pd.read_csv(ROOT / "data" / "probe_set.csv")
    y = df.label.values
    probs = {m: [] for m in bundle["models"]}
    for _, r in df.iterrows():
        t, f, fr = email_inputs(bundle, r.sender, r.subject, r.body)
        for m, p in predict_all(bundle, t, fr).items():
            probs[m].append(p)
    out = {"n": len(df), "n_phishing": int(y.sum()), "results": {m: summ(y, p) for m, p in probs.items()}}
    if args.llm:
        pl = [analyze_email(r.sender, r.subject, r.body, model=args.model)["phishing_probability"] for _, r in df.iterrows()]
        out["results"][f"LLM ({args.model})"] = summ(y, pl)
        ml = np.array(probs[bundle["best"]])
        out["results"]["Hybrid average (best ML + LLM)"] = summ(y, (ml + np.array(pl)) / 2)
    (ROOT / "results" / "probe_results.json").write_text(json.dumps(out, indent=2))
    print("| System | Acc | Prec | Recall | F1 | FPR | FNR |\n|---|---|---|---|---|---|---|")
    for k, m in out["results"].items():
        print(f"| {k} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['fpr']:.3f} | {m['fnr']:.3f} |")


if __name__ == "__main__":
    main()
