"""Explainable Phishing Email Detector - Streamlit app.

Run:  streamlit run app.py
Shows, side by side, (1) a traditional ML verdict with word/feature-level explanations and
(2) an optional LLM verdict with evidence-grounded reasoning (needs an Anthropic API key).
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from phishing_xai.explain import (describe_prob, email_inputs, highlight_html, hybrid_contributions, lime_explain,
                                  lr_contributions, predict_all, shap_explain)
from phishing_xai.features import FEATURE_DESCRIPTIONS, red_flags
from phishing_xai.llm import DEFAULT_MODEL, analyze_email

ROOT = Path(__file__).parent
st.set_page_config(page_title="Explainable Phishing Detector", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading models ...")
def load_bundle():
    candidates = [ROOT / "models" / "bundle.joblib", ROOT / "models_backup" / "bundle.joblib"]
    errors = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            bundle = joblib.load(path)
            bundle["_loaded_from"] = path.relative_to(ROOT).as_posix()
            return bundle
        except Exception as exc:  # pragma: no cover - reported to the user in the UI.
            errors.append(f"{path}: {exc}")
    raise RuntimeError(
        "The trained model bundle could not be loaded. The file is missing or corrupt. "
        "Run `python scripts/train.py` from the project root to regenerate `models/bundle.joblib`."
    )


def dataset_info(bundle: dict) -> dict:
    if bundle.get("dataset_name") and bundle.get("dataset_path"):
        return {
            "name": bundle["dataset_name"],
            "path": bundle["dataset_path"],
            "n_emails": bundle.get("dataset_n_emails"),
            "n_phishing": bundle.get("dataset_n_phishing"),
            "n_legit": bundle.get("dataset_n_legit"),
        }

    loaded_from = bundle.get("_loaded_from", "")
    if "spamassassin" in loaded_from.lower():
        metrics_path = ROOT / "results" / "metrics_spamassassin.json"
        name = "SpamAssassin"
        data_path = "data/SpamAssasin.csv"
    else:
        metrics_path = ROOT / "results" / "metrics.json"
        name = "CEAS-08"
        data_path = "data/CEAS_08.csv"

    counts = {}
    if metrics_path.exists():
        counts = json.loads(metrics_path.read_text()).get("dataset", {})
    return {
        "name": name,
        "path": data_path,
        "n_emails": counts.get("n_emails"),
        "n_phishing": counts.get("n_phishing"),
        "n_legit": counts.get("n_legit"),
    }


def load_samples() -> dict[str, dict]:
    samples = {"(write or paste your own)": {"sender": "", "subject": "", "body": ""}}
    for p in sorted((ROOT / "samples").glob("*.txt")):
        head, _, body = p.read_text().partition("\n\n")
        meta = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
        samples[p.stem.replace("_", " ").title()] = {"sender": meta.get("From", ""), "subject": meta.get("Subject", ""),
                                                      "body": body.strip()}
    return samples


def bar_chart(items, title, positive_label="pushes towards phishing"):
    if not items:
        st.info("No contributing terms found.")
        return
    items = sorted(items, key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.2, 0.34 * len(items) + 0.9))
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=["#dc3545" if v > 0 else "#198754" for _, v in items])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"red = {positive_label}, green = pushes towards legitimate", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def verdict_badge(p: float, label_hi="PHISHING / SPAM", label_lo="LEGITIMATE"):
    colour = "#dc3545" if p >= 0.5 else "#198754"
    st.markdown(f"<div style='padding:10px 14px;border-radius:8px;background:{colour};color:white;"
                f"font-size:1.25rem;font-weight:600'>{label_hi if p >= 0.5 else label_lo} "
                f"&nbsp;<span style='font-weight:400;font-size:1rem'>({p:.1%} phishing probability)</span></div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------- sidebar
try:
    bundle = load_bundle()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Settings")
    dataset = dataset_info(bundle)
    st.subheader("Active training dataset")
    st.markdown(f"**{dataset['name']}**")
    if dataset["n_emails"] is not None:
        st.caption(
            f"{dataset['n_emails']:,} emails: "
            f"{dataset['n_phishing']:,} phishing/spam, "
            f"{dataset['n_legit']:,} legitimate"
        )
    st.caption(f"Data: `{dataset['path']}`")
    st.caption(f"Model bundle: `{bundle['_loaded_from']}`")
    model_name = st.selectbox("Traditional ML model", list(bundle["models"]),
                              index=list(bundle["models"]).index(bundle["best"]))
    use_llm = st.checkbox("Also run LLM analysis", value=False)
    api_key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""),
                            disabled=not use_llm, help="Used only for this session; never stored.")
    llm_model = st.text_input("LLM model", value=DEFAULT_MODEL, disabled=not use_llm)
    lime_samples = st.slider("LIME perturbations", 200, 2000, 500, step=100,
                             help="More perturbations = more stable (but slower) LIME explanations.")
    st.caption("This is decision support, not the sole basis for blocking mail.")

st.title("🛡️ Explainable Phishing Email Detector")
st.write("Paste an email and compare a **traditional machine-learning model** with an **LLM analyst**. "
         "Both explain *why* they reached their verdict.")

samples = load_samples()
choice = st.selectbox("Load an example", list(samples))
s = samples[choice]
c1, c2 = st.columns([1, 2])
sender = c1.text_input("From", s["sender"], key=f"from_{choice}")
subject = c1.text_input("Subject", s["subject"], key=f"subj_{choice}")
body = c2.text_area("Body", s["body"], height=190, key=f"body_{choice}")

if st.button("Analyse email", type="primary", disabled=not (subject or body)):
    text, feats, feat_row = email_inputs(bundle, sender, subject, body)
    probs = predict_all(bundle, text, feat_row)
    p_ml = probs[model_name]
    kind = bundle["kinds"][model_name]

    left, right = st.columns(2, gap="large")

    # ---------------------------------------------------------------- traditional ML
    with left:
        st.subheader(f"Traditional ML: {model_name}")
        verdict_badge(p_ml)
        st.caption(f"Assessment: {describe_prob(p_ml)}")

        with st.expander("All models at a glance"):
            st.dataframe(pd.DataFrame({"P(phishing)": probs}).style.format("{:.1%}"), width="stretch")

        st.markdown("**Why?**")
        if kind == "text":
            method = st.radio("Explanation method", ["Exact coefficients", "LIME"], horizontal=True,
                              disabled=model_name != "Logistic Regression (TF-IDF)",
                              help="Exact coefficients are available for Logistic Regression only; LIME works for any text model.")
            if model_name == "Logistic Regression (TF-IDF)" and method == "Exact coefficients":
                terms, _ = lr_contributions(bundle, text, model_name, top_k=12)
                title = "Term contribution to log-odds (coefficient x TF-IDF)"
            else:
                with st.spinner("Running LIME ..."):
                    terms = lime_explain(bundle, text, model_name, num_features=12, num_samples=lime_samples)
                title = "LIME term weights"
            bar_chart(terms, title)
            st.markdown("**Email with influential words highlighted**", help="Words are shown as lower-cased tokens map to the original text.")
            st.markdown(f"<div style='font-size:0.9rem;line-height:1.6'>{highlight_html(body or subject, terms)}</div>",
                        unsafe_allow_html=True)
        elif kind == "feat":
            sv = shap_explain(bundle, model_name, feat_row, top_k=10)
            bar_chart([(f"{n} = {v:.2f}" if v != int(v) else f"{n} = {int(v)}", s_) for n, s_, v in sv],
                      "SHAP value of each feature (log-odds)")
        else:
            contrib = hybrid_contributions(bundle, model_name, text, feat_row, top_k=12)
            bar_chart(contrib, "SHAP value of each word / feature (log-odds)")
            hi = [(w, c) for w, c in contrib]
            st.markdown(f"<div style='font-size:0.9rem;line-height:1.6'>{highlight_html(body or subject, hi)}</div>",
                        unsafe_allow_html=True)

        flags = red_flags(feats)
        st.markdown("**Rule-based red flags found**")
        if flags:
            for title, detail in flags:
                st.markdown(f"- 🚩 **{title}** — {detail}")
        else:
            st.markdown("- None of the standard red flags were triggered.")
        with st.expander("Hand-crafted feature values"):
            st.dataframe(pd.DataFrame({"value": feats, "meaning": FEATURE_DESCRIPTIONS}), width="stretch")

    # ---------------------------------------------------------------- LLM
    with right:
        st.subheader("LLM analyst")
        p_llm = None
        if not use_llm:
            st.info("Tick **Also run LLM analysis** in the sidebar and enter an API key to see the LLM's verdict.")
        elif not api_key:
            st.warning("Enter an Anthropic API key in the sidebar.")
        else:
            try:
                with st.spinner("Asking the LLM ..."):
                    res = analyze_email(sender, subject, body, api_key=api_key, model=llm_model)
                p_llm = res["phishing_probability"]
                verdict_badge(p_llm)
                st.markdown(f"**Summary.** {res.get('summary', '')}")
                st.markdown(f"**Recommended action.** {res.get('recommended_action', '')}")
                st.markdown("**Indicators (with evidence check)**")
                hay = " ".join((sender + " " + subject + " " + body).lower().split())
                for ind in res["indicators"]:
                    ev = ind.get("evidence", "")
                    ok = " ".join(ev.lower().split()) in hay if ev else False
                    mark = "✅ quote found in email" if ok else "⚠️ quote NOT found verbatim"
                    st.markdown(f"- **{ind.get('category', 'other').replace('_', ' ')}** — “{ev}” \n"
                                f"  {ind.get('why', '')}  \n  <small>{mark}</small>", unsafe_allow_html=True)
                if res.get("grounding") is not None:
                    st.caption(f"Evidence grounding: {res['grounding']:.0%} of quoted snippets appear verbatim in the "
                               f"email. Model: {res['model']}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"LLM call failed: {exc}")

    # ---------------------------------------------------------------- agreement
    st.divider()
    if p_llm is not None:
        if (p_ml >= 0.5) == (p_llm >= 0.5):
            st.success(f"Both systems agree: **{'phishing / spam' if p_ml >= 0.5 else 'legitimate'}**. "
                       f"Average probability {(p_ml + p_llm) / 2:.1%}.")
        else:
            st.warning("The systems **disagree** — treat this email as suspicious and review it manually "
                       "(do not click links or open attachments until verified).")
