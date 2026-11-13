"""Explanation methods for the traditional ML models.

* Logistic Regression  -> exact per-term contributions (coefficient x TF-IDF value)
* Any text model       -> LIME (local surrogate over perturbed texts)
* Tree models on the hand-crafted features -> SHAP (TreeExplainer)
"""
from __future__ import annotations

import html
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .features import FEATURE_NAMES, clean_text, extract_features

CLASS_NAMES = ["legitimate", "phishing"]


# --------------------------------------------------------------------------- prediction helpers
def email_inputs(bundle, sender: str, subject: str, body: str):
    """Everything the models need for one email."""
    text = clean_text(subject, body)
    feats = extract_features(sender, subject, body)
    feat_row = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=float)
    return text, feats, feat_row


def _matrix(bundle, model_name, text, feat_row):
    kind = bundle["kinds"][model_name]
    if kind == "text":
        return bundle["tfidf"].transform([text])
    if kind == "feat":
        return feat_row
    sel = bundle["selector"].transform(bundle["tfidf"].transform([text]))
    return sparse.hstack([sel, sparse.csr_matrix(feat_row)]).tocsr()


def predict_all(bundle, text, feat_row) -> dict[str, float]:
    """P(phishing) from every model for a single email."""
    return {name: float(m.predict_proba(_matrix(bundle, name, text, feat_row))[0, 1])
            for name, m in bundle["models"].items()}


def text_proba_fn(bundle, model_name):
    """Callable mapping a list of cleaned texts to class probabilities (for LIME)."""
    model = bundle["models"][model_name]
    return lambda texts: model.predict_proba(bundle["tfidf"].transform(list(texts)))


# --------------------------------------------------------------------------- logistic regression
def lr_contributions(bundle, text: str, model_name="Logistic Regression (TF-IDF)", top_k=12):
    """Exact additive term contributions to the log-odds of 'phishing'."""
    model = bundle["models"][model_name]
    vec = bundle["tfidf"]
    x = vec.transform([text])
    coefs = model.coef_[0]
    names = vec.get_feature_names_out()
    idx = x.indices
    contrib = x.data * coefs[idx]
    order = np.argsort(-np.abs(contrib))[:top_k]
    rows = [(names[idx[i]], float(contrib[i])) for i in order]
    return rows, float(model.intercept_[0])


# --------------------------------------------------------------------------- LIME
def lime_explain(bundle, text: str, model_name="Logistic Regression (TF-IDF)", num_features=10,
                 num_samples=500, seed=0):
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=seed, bow=True)
    exp = explainer.explain_instance(text, text_proba_fn(bundle, model_name), num_features=num_features,
                                     num_samples=num_samples, labels=(1,))
    return [(w, float(v)) for w, v in exp.as_list(label=1)]


# --------------------------------------------------------------------------- SHAP (hand-crafted features)
_SHAP_CACHE: dict = {}


def shap_explain(bundle, model_name: str, feat_row: np.ndarray, top_k=8):
    import shap
    model = bundle["models"][model_name]
    if model_name not in _SHAP_CACHE:
        _SHAP_CACHE[model_name] = shap.TreeExplainer(model)
    sv = _SHAP_CACHE[model_name].shap_values(feat_row)
    if isinstance(sv, list):          # older SHAP: list of per-class arrays
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        sv = sv[:, :, 1]
    vals = sv[0]
    order = np.argsort(-np.abs(vals))[:top_k]
    return [(FEATURE_NAMES[i], float(vals[i]), float(feat_row[0, i])) for i in order]


# --------------------------------------------------------------------------- display
def highlight_html(original: str, weights: Iterable[tuple[str, float]], max_chars=3000) -> str:
    """Return HTML where words with positive weight (towards phishing) are red, negative are green."""
    wmap = {}
    for term, w in weights:
        for tok in term.split():
            wmap[tok.lower()] = wmap.get(tok.lower(), 0) + w
    if not wmap:
        return html.escape(original[:max_chars])
    scale = max(abs(v) for v in wmap.values()) or 1.0
    out = []
    for piece in re.split(r"(\W+)", original[:max_chars]):
        key = piece.lower()
        if key in wmap and re.match(r"\w", piece):
            v = wmap[key] / scale
            a = min(0.15 + 0.6 * abs(v), 0.75)
            colour = f"rgba(220,53,69,{a:.2f})" if v > 0 else f"rgba(25,135,84,{a:.2f})"
            out.append(f"<span style='background:{colour};border-radius:3px;padding:0 2px'>{html.escape(piece)}</span>")
        else:
            out.append(html.escape(piece))
    return "".join(out).replace("\n", "<br>")


def describe_prob(p: float) -> str:
    if p >= 0.9:
        return "very likely phishing/spam"
    if p >= 0.6:
        return "likely phishing/spam"
    if p > 0.4:
        return "uncertain"
    if p > 0.1:
        return "likely legitimate"
    return "very likely legitimate"


# --------------------------------------------------------------------------- XGBoost hybrid (TF-IDF + features)
def hybrid_contributions(bundle, model_name: str, text: str, feat_row: np.ndarray, top_k=12):
    """Exact TreeSHAP values from XGBoost's built-in `pred_contribs` for the TF-IDF + handcrafted model."""
    import xgboost as xgb
    model = bundle["models"][model_name]
    X = _matrix(bundle, model_name, text, feat_row)
    names = list(bundle["tfidf"].get_feature_names_out()[bundle["selector"].get_support()]) + list(FEATURE_NAMES)
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]   # last = bias
    order = np.argsort(-np.abs(contribs))[:top_k]
    return [(names[i], float(contribs[i])) for i in order if contribs[i] != 0]
