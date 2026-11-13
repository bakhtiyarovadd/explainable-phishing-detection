"""Convert raw email files into the CSV format used by this project (sender, subject, body, label).

Examples
  python scripts/prepare_external.py --mbox phishing.mbox --label 1 --out data/external_phish.csv
  python scripts/prepare_external.py --dir easy_ham/ --label 0 --out data/external_ham.csv
Works with .mbox files (e.g. the Nazario phishing corpus) and folders with one raw email per file
(e.g. the SpamAssassin public corpus). Run it once per class, then combine the CSVs (see README).
"""
import argparse
import mailbox
from email import message_from_binary_file, policy
from pathlib import Path

import pandas as pd


def body_of(msg) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    try:
        return part.get_content()
    except Exception:  # noqa: BLE001  (bad encodings in old mail)
        payload = part.get_payload(decode=True) or b""
        return payload.decode("utf-8", errors="replace")


def row(msg, label):
    return {"sender": str(msg.get("From", "")), "subject": str(msg.get("Subject", "")), "body": body_of(msg), "label": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbox")
    ap.add_argument("--dir")
    ap.add_argument("--label", type=int, required=True, choices=[0, 1], help="1 = phishing/spam, 0 = legitimate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    if a.mbox:
        for m in mailbox.mbox(a.mbox, factory=lambda f: message_from_binary_file(f, policy=policy.default)):
            rows.append(row(m, a.label))
    if a.dir:
        for p in sorted(Path(a.dir).rglob("*")):
            if p.is_file() and not p.name.startswith("cmds"):
                with open(p, "rb") as f:
                    rows.append(row(message_from_binary_file(f, policy=policy.default), a.label))
    df = pd.DataFrame(rows)
    before = len(df)
    df = df[df["body"].fillna("").str.strip().str.len() > 0].drop_duplicates(subset=["subject", "body"])
    df.to_csv(a.out, index=False)
    print(f"Read {before} emails, kept {len(df)} (removed empty bodies and duplicates). Saved {a.out}")


if __name__ == "__main__":
    main()
