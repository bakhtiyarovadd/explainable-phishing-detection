# Explainable Phishing Email Detection (LLM + traditional ML)

NTCC minor project. A Streamlit app that classifies an email as phishing/spam or legitimate with six traditional
ML models and, optionally, an LLM analyst, and **explains** every decision.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py                 # trained models are included in models/
```
To enable the LLM analyst, tick the box in the sidebar and paste an Anthropic API key (or `export ANTHROPIC_API_KEY=...`).
The model name can be changed in the sidebar or with `PHISHING_LLM_MODEL`.

If the app shows `input stream corrupted` while loading the model bundle, the pickled bundle on disk is stale or corrupted.
Run `python scripts/train.py` to regenerate `models/bundle.joblib`.

## Reproduce the experiments
```bash
python scripts/get_data.py                 # needed for training and the LLM / explanation evaluations
python scripts/train.py                    # ~4 min on one CPU core; writes models/ and results/metrics.json
python scripts/evaluate_probe.py           # 24-email modern probe set (domain shift)
python scripts/evaluate_explanations.py    # deletion test of LR coefficients vs LIME vs random
python scripts/make_figures.py             # all figures in results/figures/
```

## Complete the LLM evaluation (needs an API key)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/evaluate_llm.py --n-per-class 100     # 200 API calls; prints a ready-to-paste table
python scripts/evaluate_probe.py --llm               # LLM + hybrid on the probe set
```
Results are cached in `results/llm_cache.jsonl`; re-running resumes where it stopped. Paste the printed tables into
Table 7 of the report and Table III of the paper, then write the short discussion paragraph.

## Layout
| Path | Content |
|---|---|
| `app.py` | Streamlit application |
| `phishing_xai/features.py` | cleaning + 25 red-flag features |
| `phishing_xai/explain.py` | coefficients, LIME, SHAP, highlighting |
| `phishing_xai/llm.py` | LLM analyst, JSON parsing, evidence-grounding check |
| `scripts/` | training and evaluation |
| `models/bundle.joblib` | trained vectoriser + 6 models |
| `results/` | metrics, figures, logs |
| `data/probe_set.csv` | 24 hand-written modern emails (12 phishing, 12 legitimate) |
| `samples/` | demo emails for the app |

## Notes
* Dataset: CEAS-08 as redistributed at github.com/PHP4Life/Phishing-Email-Dataset. Its positive class is mostly classic
  spam, so test accuracy (~99.8%) overstates real-world performance; see the report, Section 4.5.
* Use as decision support only. Do not paste sensitive real emails into the LLM option without checking your privacy policy.
