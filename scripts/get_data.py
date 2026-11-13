"""Download the CEAS-08 corpus into data/CEAS_08.csv (about 67 MB)."""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/PHP4Life/Phishing-Email-Dataset/main/CEAS_08.csv"
dest = Path(__file__).resolve().parents[1] / "data" / "CEAS_08.csv"
dest.parent.mkdir(exist_ok=True)
print("Downloading", URL)
urllib.request.urlretrieve(URL, dest)
print("Saved", dest, f"({dest.stat().st_size / 1e6:.1f} MB)")
