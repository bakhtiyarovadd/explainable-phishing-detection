"""Quantify how faithful the text explanations are (deletion / 'comprehensiveness' test).

For each test email, the top-k terms identified by an explanation method are deleted from the text and the
drop in the model's probability for its originally predicted class is measured. A faithful explanation should
cause a much larger drop than deleting k random words.

Methods compared on the Logistic Regression model: exact coefficient contributions, LIME, random words.
Usage: python scripts/evaluate_explanations.py [--n 200]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phishing_xai.explain import lime_explain, lr_contributions, text_proba_fn  # noqa: E402
from phishing_xai.features import clean_text  # noqa: E402

MODEL = "Logistic Regression (TF-IDF)"


def delete_terms(text: str, terms) -> str:
    words = set()
    for t in terms:
        words.update(t.split())
    return " ".join(w for w in text.split() if re.sub(r"\W", "", w) not in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    bundle = joblib.load(ROOT / "models" / "bundle.joblib")
    df = pd.read_csv(ROOT / "data" / "CEAS_08.csv")
    df["row_id"] = np.arange(len(df))
    test = pd.read_csv(ROOT / "results" / "test_ids.csv")
    per = args.n // 2
    ids = pd.concat([test[test.label == c].sample(per, random_state=args.seed) for c in (0, 1)]).row_id.values
    df = df.set_index("row_id").loc[ids]
    prob = text_proba_fn(bundle, MODEL)

    ks = (3, 5, 10)
    drops = {m: {k: [] for k in ks} for m in ("coefficients", "lime", "random")}
    t_coef = t_lime = 0.0
    for _, e in df.iterrows():
        text = clean_text(e["subject"], e["body"])
        base = prob([text])[0]
        cls = int(base[1] >= 0.5)
        p0 = base[cls]

        t = time.time()
        rows, _ = lr_contributions(bundle, text, MODEL, top_k=max(ks) * 3)
        t_coef += time.time() - t
        sign = 1 if cls == 1 else -1
        coef_terms = [w for w, c in rows if c * sign > 0]

        t = time.time()
        lime_terms = [w for w, c in lime_explain(bundle, text, MODEL, num_features=max(ks) * 2, num_samples=500)
                      if c * sign > 0]
        t_lime += time.time() - t

        vocab = [w for w in set(text.split()) if len(w) > 2]
        for k in ks:
            rand_terms = list(rng.choice(vocab, size=min(k, len(vocab)), replace=False)) if vocab else []
            for name, terms in (("coefficients", coef_terms[:k]), ("lime", lime_terms[:k]), ("random", rand_terms)):
                p1 = prob([delete_terms(text, terms)])[0][cls]
                drops[name][k].append(p0 - p1)

    n = len(df)
    out = {"n": n, "model": MODEL,
           "mean_probability_drop": {m: {str(k): float(np.mean(v)) for k, v in d.items()} for m, d in drops.items()},
           "seconds_per_explanation": {"coefficients": t_coef / n, "lime": t_lime / n}}
    (ROOT / "results" / "explanation_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
