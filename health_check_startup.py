"""
health_check_startup.py
Lightweight startup health check called by start_dashboard.bat.
Separate file avoids Windows CMD multi-line python -c quoting problems.
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

cache_dir = Path(".cache")
if not cache_dir.exists():
    print("  WARNING: Cache not found -- first run fetches all data (~3 min)")
    sys.exit(0)

checks = [
    ("fred_T10Y2Y.parquet",       "Yield Curve",    1),
    ("fred_BAMLH0A0HYM2.parquet", "HY Spreads",     1),
    ("shiller_ie_data.parquet",   "Shiller CAPE",  30),
    ("aaii_sentiment.parquet",    "AAII Sentiment",  7),
    ("fred_NFCI.parquet",         "NFCI",            7),
    ("fred_IPMAN.parquet",        "ISM PMI (proxy)", 30),
]
for filename, label, max_days in checks:
    path = cache_dir / filename
    if not path.exists():
        print(f"  MISSING {label}: not cached (run: python data_fetcher.py)")
    else:
        age = (datetime.utcnow() -
               datetime.utcfromtimestamp(path.stat().st_mtime)).days
        status = "OK" if age <= max_days else "STALE"
        print(f"  {status}     {label}: {age}d old")

# Check for trained models in .cache/models/
models_dir = cache_dir / "models"
model_files = list(models_dir.glob("*.joblib")) if models_dir.exists() else []
if model_files:
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    age_days = (datetime.utcnow() -
                datetime.utcfromtimestamp(latest.stat().st_mtime)).days
    print(f"  OK     ML models: {len(model_files)} found, newest {age_days}d old")
else:
    print("  MISSING ML models: not trained (run: python ml_model.py)")

# Current risk score
try:
    from bear_risk_api import get_current_risk_score
    result = get_current_risk_score()
    if result:
        score = result["score"]
        band  = result["band"]
        bar   = chr(9608) * int(score / 5) + chr(9617) * (20 - int(score / 5))
        print(f"\n  Bear Risk Score: {score:.0f}/100  [{bar}]  {band}")
    else:
        print("\n  Bear Risk Score: N/A (cache empty)")
except Exception as e:
    print(f"\n  Score unavailable: {e}")
