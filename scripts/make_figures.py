"""Generate all figures used in the report/paper from results/*.json and the trained models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10})
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#2b5c8a", "#c0392b", "#2e8b57", "#7f8c8d", "#e08e0b", "#6c4f9e"
SHORT = {"Logistic Regression (TF-IDF)": "LR\n(TF-IDF)", "Linear SVM (TF-IDF)": "SVM\n(TF-IDF)",
         "Naive Bayes (TF-IDF)": "NB\n(TF-IDF)", "Random Forest (handcrafted)": "RF\n(handcrafted)",
         "XGBoost (handcrafted)": "XGB\n(handcrafted)", "XGBoost (TF-IDF + handcrafted)": "XGB\n(hybrid)"}


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def box(ax, x, y, w, h, text, fc, ec=None, fs=8.5, bold=False, tc="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec or fc, lw=1.2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.3, color=color))


def architecture():
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 6.6); ax.axis("off")
    # stage 0: input
    box(ax, 0.1, 2.7, 1.9, 1.3, "Incoming email\n(sender, subject,\nbody)", "#ecf0f1", GREY, fs=8, bold=True)
    # stage 1: preprocessing
    ax.text(3.7, 6.3, "Traditional ML branch", ha="center", fontsize=9.5, fontweight="bold", color=BLUE)
    box(ax, 2.5, 4.35, 2.4, 1.3, "Text cleaning\nHTML strip, URL token,\nTF-IDF (1-2 grams)", "#dbe9f6", BLUE, fs=7.4)
    box(ax, 2.5, 2.2, 2.4, 1.3, "Feature extraction\n25 interpretable\nred-flag features", "#dbe9f6", BLUE, fs=8)
    # stage 2: models
    box(ax, 5.6, 4.95, 2.7, 0.95, "LR / Linear SVM / NB\n(text models)", "#c9dff3", BLUE, fs=8)
    box(ax, 5.6, 3.65, 2.7, 0.95, "XGBoost hybrid\n(TF-IDF + features)", "#c9dff3", BLUE, fs=8)
    box(ax, 5.6, 2.35, 2.7, 0.95, "Random Forest / XGBoost\n(handcrafted features)", "#c9dff3", BLUE, fs=7.6)
    # stage 3: explainers
    box(ax, 8.9, 4.95, 2.2, 0.95, "Coefficients\n+ LIME", "#d6f0e0", GREEN, fs=8)
    box(ax, 8.9, 3.65, 2.2, 0.95, "TreeSHAP\n(words + features)", "#d6f0e0", GREEN, fs=8)
    box(ax, 8.9, 2.35, 2.2, 0.95, "TreeSHAP on\nred-flag features", "#d6f0e0", GREEN, fs=8)
    # LLM branch
    ax.text(3.7, 1.35, "LLM branch", ha="center", fontsize=9.5, fontweight="bold", color=PURPLE)
    box(ax, 5.6, 0.1, 2.7, 1.2, "LLM analyst\nhardened security prompt\n(email = untrusted data)", "#e6dcf3", PURPLE, fs=7.2)
    box(ax, 8.9, 0.1, 2.2, 1.2, "JSON verdict,\nquoted indicators,\ngrounding check", "#e6dcf3", PURPLE, fs=7.8)
    # fusion
    box(ax, 11.7, 1.5, 1.8, 3.4, "Fusion & UI\n\nverdict,\nagreement\ncheck,\nhighlighted\nevidence", "#fdebd0", ORANGE, fs=8, bold=True)
    # arrows
    arrow(ax, 2.0, 3.6, 2.5, 4.9); arrow(ax, 2.0, 3.2, 2.5, 2.9)
    arrow(ax, 4.9, 5.2, 5.6, 5.4); arrow(ax, 4.9, 4.8, 5.6, 4.2)
    arrow(ax, 4.9, 3.0, 5.6, 2.85); arrow(ax, 4.9, 3.3, 5.6, 4.0)
    for y in (5.42, 4.12, 2.82):
        arrow(ax, 8.3, y, 8.9, y)
    arrow(ax, 11.1, 5.4, 11.7, 4.5); arrow(ax, 11.1, 4.12, 11.7, 3.7); arrow(ax, 11.1, 2.82, 11.7, 3.0)
    # LLM route: down from input, then right
    ax.plot([1.05, 1.05], [2.7, 0.7], color=PURPLE, lw=1.3)
    arrow(ax, 1.05, 0.7, 5.6, 0.7, PURPLE)
    arrow(ax, 8.3, 0.7, 8.9, 0.7, PURPLE)
    ax.plot([11.1, 11.4], [0.7, 0.7], color=PURPLE, lw=1.3); ax.plot([11.4, 11.4], [0.7, 2.2], color=PURPLE, lw=1.3)
    arrow(ax, 11.4, 2.2, 11.7, 2.2, PURPLE)
    save(fig, "fig1_architecture.png")


def class_distribution(m):
    d = m["dataset"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    vals = [d["n_legit"], d["n_phishing"]]
    b = ax.bar(["Legitimate", "Phishing / spam"], vals, color=[GREEN, RED], width=0.55)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 300, f"{v:,}\n({v / d['n_emails']:.1%})", ha="center", fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.25); ax.set_ylabel("Number of emails"); ax.set_title("CEAS-08 class distribution")
    save(fig, "fig2_class_distribution.png")


def model_comparison(m, probe):
    names = list(m["models"])
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    x = np.arange(len(names)); w = 0.38
    a = [m["models"][n]["accuracy"] * 100 for n in names]
    p = [probe["results"][n]["accuracy"] * 100 for n in names]
    b1 = ax.bar(x - w / 2, a, w, color=BLUE, label="CEAS-08 test set (n = 7,831)")
    b2 = ax.bar(x + w / 2, p, w, color=ORANGE, label="Modern-email probe set (n = 24)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], fontsize=8)
    ax.set_ylim(0, 112); ax.set_ylabel("Accuracy (%)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    ax.set_title("In-distribution accuracy vs. accuracy on modern emails")
    save(fig, "fig3_domain_shift.png")


def roc(m):
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
    cols = [BLUE, "#5b9bd5", GREY, RED, ORANGE, GREEN]
    for (n, c), col in zip(m["roc"].items(), cols):
        for ax in axs:
            ax.plot(c["fpr"], c["tpr"], color=col, lw=1.6, label=f"{SHORT[n].replace(chr(10), ' ')}  (AUC {m['models'][n]['roc_auc']:.4f})")
    axs[0].plot([0, 1], [0, 1], "--", color="#bbb", lw=1)
    axs[0].set_title("ROC curves"); axs[1].set_title("Zoom: low false-positive region")
    axs[1].set_xlim(0, 0.05); axs[1].set_ylim(0.94, 1.002)
    for ax in axs:
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    axs[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_roc.png")


def confusion(m):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, n in zip(axs, ["Linear SVM (TF-IDF)", "Random Forest (handcrafted)"]):
        r = m["models"][n]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Phish"]); ax.set_yticklabels(["Legit", "Phish"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(SHORT[n].replace("\n", " "), fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, "fig5_confusion.png")


def lr_terms(bundle):
    lr = bundle["models"]["Logistic Regression (TF-IDF)"]
    names = bundle["tfidf"].get_feature_names_out()
    o = np.argsort(lr.coef_[0])
    top_leg, top_ph = o[:12], o[::-1][:12]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axs[0].barh([names[i] for i in top_ph][::-1], lr.coef_[0][top_ph][::-1], color=RED)
    axs[0].set_title("Terms pushing towards phishing / spam")
    axs[1].barh([names[i] for i in top_leg][::-1], -lr.coef_[0][top_leg][::-1], color=GREEN)
    axs[1].set_title("Terms pushing towards legitimate"); axs[1].set_xlabel("|coefficient|"); axs[0].set_xlabel("coefficient")
    fig.tight_layout()
    save(fig, "fig6_lr_global_terms.png")


def local_explanation(bundle):
    from phishing_xai.explain import email_inputs, lr_contributions
    p = (ROOT / "samples" / "phishing_account_suspended.txt").read_text()
    head, _, body = p.partition("\n\n")
    meta = dict(l.split(": ", 1) for l in head.splitlines())
    text, _, _ = email_inputs(bundle, meta["From"], meta["Subject"], body)
    rows, _ = lr_contributions(bundle, text, top_k=12)
    rows = sorted(rows, key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.barh([k for k, _ in rows], [v for _, v in rows], color=[RED if v > 0 else GREEN for _, v in rows])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Contribution to log-odds of phishing"); ax.set_title("Local explanation (LR) for a sample email")
    fig.tight_layout()
    save(fig, "fig7_local_explanation.png")


def shap_global(bundle):
    import shap
    F = pd.read_csv(ROOT / "results" / "test_features.csv")
    X = F.drop(columns=["label"]).sample(1500, random_state=0)
    model = bundle["models"]["XGBoost (handcrafted)"]
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X.values))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = np.abs(sv).mean(0)
    o = np.argsort(imp)[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.barh([X.columns[i] for i in o], imp[o], color=BLUE)
    ax.set_xlabel("Mean |SHAP value| (log-odds)"); ax.set_title("Global importance: XGBoost on handcrafted features")
    fig.tight_layout()
    save(fig, "fig8_shap_global.png")
    json.dump({X.columns[i]: float(imp[i]) for i in np.argsort(imp)[::-1]}, open(ROOT / "results" / "shap_global.json", "w"), indent=2)


def faithfulness():
    e = json.load(open(ROOT / "results" / "explanation_eval.json"))["mean_probability_drop"]
    ks = ["3", "5", "10"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(ks)); w = 0.26
    for i, (k, lab, col) in enumerate([("coefficients", "LR coefficients", BLUE), ("lime", "LIME", PURPLE), ("random", "Random words", GREY)]):
        ax.bar(x + (i - 1) * w, [e[k][t] for t in ks], w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([f"k = {k}" for k in ks]); ax.set_ylabel("Mean drop in predicted-class probability")
    ax.set_title("Deletion test (comprehensiveness)"); ax.legend(frameon=False, fontsize=8)
    save(fig, "fig9_faithfulness.png")


if __name__ == "__main__":
    m = json.load(open(ROOT / "results" / "metrics.json"))
    probe = json.load(open(ROOT / "results" / "probe_results.json"))
    bundle = joblib.load(ROOT / "models" / "bundle.joblib")
    architecture(); class_distribution(m); model_comparison(m, probe); roc(m); confusion(m)
    lr_terms(bundle); local_explanation(bundle); shap_global(bundle); faithfulness()
