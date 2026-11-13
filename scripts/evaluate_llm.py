"""Evaluate the LLM detector (and a hybrid ML+LLM score) on a stratified sample of the held-out test set.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/evaluate_llm.py --n-per-class 100          # 200 API calls
    python scripts/evaluate_llm.py --n-per-class 100 --model claude-sonnet-5-5

Results are cached in results/llm_cache.jsonl, so an interrupted run can simply be restarted.
Outputs: results/llm_results.json and a ready-to-paste Markdown table printed to the terminal.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phishing_xai.explain import email_inputs, predict_all  # noqa: E402
from phishing_xai.llm import DEFAULT_MODEL, analyze_email  # noqa: E402


def summarise(y, p, thr=0.5):
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred), "f1": f1_score(y, pred), "fpr": fp / max(fp + tn, 1),
           "fnr": fn / max(fn + tp, 1), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    try:
        out["roc_auc"] = roc_auc_score(y, p)
    except ValueError:
        out["roc_auc"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(ROOT / "data" / "CEAS_08.csv"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    bundle = joblib.load(ROOT / "models" / "bundle.joblib")
    ml_name = bundle["best"]
    df = pd.read_csv(args.data)
    df["row_id"] = np.arange(len(df))
    test = pd.read_csv(ROOT / "results" / "test_ids.csv")
    sample = pd.concat([test[test.label == c].sample(args.n_per_class, random_state=args.seed) for c in (0, 1)])
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df = df.set_index("row_id")

    cache_path = ROOT / "results" / "llm_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["row_id"], r["model"])] = r

    rows = []
    for i, rec in sample.iterrows():
        rid = int(rec.row_id)
        e = df.loc[rid]
        key = (rid, args.model)
        if key not in cache:
            for attempt in range(3):
                try:
                    res = analyze_email(e["sender"], e["subject"], e["body"], model=args.model)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  retry {attempt + 1} for row {rid}: {exc}")
                    time.sleep(3 * (attempt + 1))
            else:
                continue
            res.update(row_id=rid, model=args.model)
            cache[key] = res
            with cache_path.open("a") as f:
                f.write(json.dumps(res) + "\n")
        res = cache[key]
        text, feats, feat_row = email_inputs(bundle, e["sender"], e["subject"], e["body"])
        probs = predict_all(bundle, text, feat_row)
        rows.append({"row_id": rid, "y": int(rec.label), "p_llm": res["phishing_probability"],
                     "p_ml": probs[ml_name], "grounding": res.get("grounding"),
                     "n_indicators": len(res.get("indicators", []))})
        if len(rows) % 20 == 0:
            print(f"  {len(rows)}/{len(sample)} emails analysed")

    R = pd.DataFrame(rows)
    y = R.y.values
    R["p_hybrid_avg"] = (R.p_ml + R.p_llm) / 2
    R["p_hybrid_either"] = np.maximum(R.p_ml, R.p_llm)
    table = {
        f"{ml_name} (ML only)": summarise(y, R.p_ml),
        f"LLM only ({args.model})": summarise(y, R.p_llm),
        "Hybrid: average of ML and LLM": summarise(y, R.p_hybrid_avg),
        "Hybrid: flag if either flags": summarise(y, R.p_hybrid_either),
    }
    disagree = ((R.p_ml >= 0.5) != (R.p_llm >= 0.5)).mean()
    out = {"model": args.model, "n": int(len(R)), "ml_model": ml_name, "results": table,
           "disagreement_rate": float(disagree),
           "mean_grounding_rate": float(R.grounding.dropna().mean()) if R.grounding.notna().any() else None,
           "mean_indicators": float(R.n_indicators.mean())}
    (ROOT / "results" / "llm_results.json").write_text(json.dumps(out, indent=2))
    R.to_csv(ROOT / "results" / "llm_predictions.csv", index=False)

    print(f"\nEvaluated {len(R)} emails ({args.n_per_class} per class)\n")
    print("| System | Accuracy | Precision | Recall | F1 | FPR | FNR |\n|---|---|---|---|---|---|---|")
    for k, m in table.items():
        print(f"| {k} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
              f"{m['fpr']:.3f} | {m['fnr']:.3f} |")
    print(f"\nML/LLM disagreement rate: {disagree:.1%}")
    print(f"Mean evidence-grounding rate: {out['mean_grounding_rate']}")


if __name__ == "__main__":
    main()
