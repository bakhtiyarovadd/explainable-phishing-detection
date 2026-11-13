"""Text cleaning and hand-crafted, human-readable features.

The hand-crafted features are deliberately interpretable: every one of them
corresponds to a red flag a security analyst would recognise, which makes the
tree-based models (Random Forest / XGBoost) explainable with SHAP.
"""
from __future__ import annotations

import html
import math
import re
from email.utils import parseaddr

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+", re.I)
TAG_RE = re.compile(r"<[^>]+>")
HREF_RE = re.compile(r"<a\s[^>]*href", re.I)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.I)
WORD_RE = re.compile(r"[A-Za-z']+")

SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly")
SUSPICIOUS_TLDS = (".ru", ".cn", ".tk", ".top", ".xyz", ".info", ".biz", ".click", ".work", ".gq", ".ml", ".cf")
FREE_MAIL = ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "mail.ru", "yandex.ru")

URGENCY = ("urgent", "immediately", "act now", "asap", "expires", "expire", "within 24", "limited time",
           "final notice", "last chance", "suspended", "suspend", "verify now", "deadline", "right away")
CREDENTIAL = ("password", "login", "log in", "sign in", "username", "account number", "pin ", "ssn",
              "social security", "credit card", "card number", "verify your", "confirm your", "update your",
              "security question", "one-time", "otp")
MONEY = ("winner", "won ", "prize", "lottery", "claim", "refund", "$", "usd", "million", "inheritance",
         "cash", "bonus", "free", "discount", "offer", "payment", "invoice", "wire transfer")
GREETING_GENERIC = ("dear customer", "dear user", "dear sir", "dear member", "dear client", "valued customer",
                    "dear account holder", "dear friend")

FEATURE_NAMES = [
    "url_count", "url_len_mean", "has_ip_url", "has_shortener", "suspicious_tld_count",
    "num_html_links", "is_html", "urgency_hits", "credential_hits", "money_hits",
    "generic_greeting", "exclaim_count", "question_count", "upper_word_ratio", "digit_ratio",
    "non_ascii_ratio", "subject_len", "subject_upper_ratio", "subject_has_re_fwd",
    "log_body_len", "word_count_log", "sender_digit_ratio", "sender_free_mail",
    "sender_name_domain_mismatch", "sender_has_display_name",
]

# Plain-language descriptions used in the UI.
FEATURE_DESCRIPTIONS = {
    "url_count": "Number of links in the message",
    "url_len_mean": "Average link length (long links often hide redirects)",
    "has_ip_url": "A link points to a raw IP address",
    "has_shortener": "A link uses a URL-shortening service",
    "suspicious_tld_count": "Links ending in a high-abuse domain suffix",
    "num_html_links": "Number of clickable HTML links",
    "is_html": "Message is formatted as HTML",
    "urgency_hits": "Urgency / pressure phrases (e.g. 'act now', 'suspended')",
    "credential_hits": "Requests for passwords, card numbers or account verification",
    "money_hits": "Money, prize or offer language",
    "generic_greeting": "Generic greeting such as 'Dear customer'",
    "exclaim_count": "Exclamation marks",
    "question_count": "Question marks",
    "upper_word_ratio": "Share of ALL-CAPS words",
    "digit_ratio": "Share of digits in the text",
    "non_ascii_ratio": "Share of unusual (non-ASCII) characters",
    "subject_len": "Subject length",
    "subject_upper_ratio": "Share of capital letters in the subject",
    "subject_has_re_fwd": "Subject starts with Re:/Fwd: (reply-thread style)",
    "log_body_len": "Message length (log scale)",
    "word_count_log": "Word count (log scale)",
    "sender_digit_ratio": "Digits in the sender address",
    "sender_free_mail": "Sender uses a free mail provider",
    "sender_name_domain_mismatch": "Display name does not match sender domain",
    "sender_has_display_name": "Sender has a display name",
}


def _s(x) -> str:
    """Coerce a possibly-missing cell to a string."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = TAG_RE.sub(" ", text)
    return html.unescape(text)


def clean_text(subject: str, body: str, max_chars: int = 5000) -> str:
    """Normalised text for TF-IDF: HTML removed, URLs/addresses replaced by tokens."""
    text = f"{_s(subject)} . {strip_html(_s(body)[: max_chars * 2])}"
    text = URL_RE.sub(" urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text[:max_chars]


def _count(haystack: str, needles) -> int:
    return sum(haystack.count(n) for n in needles)


def extract_features(sender: str, subject: str, body: str) -> dict:
    sender, subject, body = _s(sender), _s(subject), _s(body)
    raw = body[:20000]
    plain = strip_html(raw)
    low = (subject + " " + plain).lower()
    urls = URL_RE.findall(raw)
    url_hosts = [re.sub(r"^https?://", "", u, flags=re.I).split("/")[0].lower() for u in urls]

    words = WORD_RE.findall(plain)
    n_words = max(len(words), 1)
    upper_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    chars = max(len(plain), 1)

    name, addr = parseaddr(sender)
    domain = addr.split("@")[-1].lower() if "@" in addr else ""
    local = addr.split("@")[0] if "@" in addr else addr
    dom_core = domain.split(".")[0] if domain else ""
    name_l = re.sub(r"[^a-z]", "", name.lower())
    impersonation = ("bank", "paypal", "amazon", "apple", "microsoft", "google", "support", "security",
                     "service", "admin")
    mismatch = int(bool(name_l) and len(dom_core) > 3 and dom_core not in name_l and name_l not in dom_core
                   and any(b in name_l for b in impersonation))

    return {
        "url_count": len(urls),
        "url_len_mean": float(np.mean([len(u) for u in urls])) if urls else 0.0,
        "has_ip_url": int(bool(IP_URL_RE.search(raw))),
        "has_shortener": int(any(h in SHORTENERS for h in url_hosts)),
        "suspicious_tld_count": sum(h.endswith(SUSPICIOUS_TLDS) for h in url_hosts),
        "num_html_links": len(HREF_RE.findall(raw)),
        "is_html": int(bool(re.search(r"<(html|body|table|div|a)\b", raw, re.I))),
        "urgency_hits": _count(low, URGENCY),
        "credential_hits": _count(low, CREDENTIAL),
        "money_hits": _count(low, MONEY),
        "generic_greeting": int(any(g in low for g in GREETING_GENERIC)),
        "exclaim_count": plain.count("!"),
        "question_count": plain.count("?"),
        "upper_word_ratio": upper_words / n_words,
        "digit_ratio": sum(c.isdigit() for c in plain) / chars,
        "non_ascii_ratio": sum(ord(c) > 127 for c in plain) / chars,
        "subject_len": len(subject),
        "subject_upper_ratio": sum(c.isupper() for c in subject) / max(len(subject), 1),
        "subject_has_re_fwd": int(bool(re.match(r"\s*(re|fwd?)\s*:", subject, re.I))),
        "log_body_len": math.log1p(len(plain)),
        "word_count_log": math.log1p(len(words)),
        "sender_digit_ratio": sum(c.isdigit() for c in local) / max(len(local), 1),
        "sender_free_mail": int(domain in FREE_MAIL),
        "sender_name_domain_mismatch": mismatch,
        "sender_has_display_name": int(bool(name.strip())),
    }


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [extract_features(s, sub, b) for s, sub, b in zip(df["sender"], df["subject"], df["body"])]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def red_flags(feats: dict) -> list[tuple[str, str]]:
    """Human-readable red flags (title, detail) triggered by a feature dict."""
    flags = []
    if feats["has_ip_url"]:
        flags.append(("Link to a raw IP address", FEATURE_DESCRIPTIONS["has_ip_url"]))
    if feats["has_shortener"]:
        flags.append(("URL shortener", FEATURE_DESCRIPTIONS["has_shortener"]))
    if feats["suspicious_tld_count"]:
        flags.append(("High-abuse domain suffix", f"{feats['suspicious_tld_count']} link(s) use one"))
    if feats["urgency_hits"]:
        flags.append(("Urgency language", f"{feats['urgency_hits']} pressure phrase(s) found"))
    if feats["credential_hits"]:
        flags.append(("Credential / payment request", f"{feats['credential_hits']} phrase(s) found"))
    if feats["money_hits"] >= 3:
        flags.append(("Money / prize language", f"{feats['money_hits']} phrase(s) found"))
    if feats["generic_greeting"]:
        flags.append(("Generic greeting", FEATURE_DESCRIPTIONS["generic_greeting"]))
    if feats["upper_word_ratio"] > 0.1:
        flags.append(("Excessive ALL-CAPS", f"{feats['upper_word_ratio']:.0%} of words"))
    if feats["exclaim_count"] >= 3:
        flags.append(("Many exclamation marks", f"{feats['exclaim_count']} found"))
    if feats["sender_name_domain_mismatch"]:
        flags.append(("Sender name / domain mismatch", FEATURE_DESCRIPTIONS["sender_name_domain_mismatch"]))
    if feats["url_count"] >= 5:
        flags.append(("Many links", f"{feats['url_count']} links"))
    return flags
