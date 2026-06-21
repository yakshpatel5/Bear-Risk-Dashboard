#!/usr/bin/env python3
"""
patch_app.py  --  fixes the app.py f-string crash and HorizonModel load error.
Run ONCE from your project folder:
    python patch_app.py
"""
import re, shutil, sys
from pathlib import Path

app_path = Path("app.py")
if not app_path.exists():
    print("ERROR: app.py not found. Run from your project folder.")
    sys.exit(1)

shutil.copy(app_path, "app.py.bak")
src = app_path.read_text(encoding="utf-8")
fixes = 0

# ------------------------------------------------------------------
# FIX 1: Invalid f-string format spec (the crash on line 1068)
# Python does NOT allow: f"{x:.0f if cond else 'y'}"
# Must pre-compute the formatted string before the f-string.
# ------------------------------------------------------------------
OLD_LINE = (
    "with st.expander("
    "f\"{row['name'][:22]}  \u00b7  "
    "{score:.0f if not pd.isna(score) else '\u2014'}\"):"
)
NEW_LINE = (
    "_score_str = (f\"{score:.0f}\" "
    "if (score is not None and not __import__('pandas').isna(score)) "
    "else \"-\")\n"
    "        with st.expander("
    "f\"{row['name'][:22]}  \u00b7  {_score_str}\"):"
)
if OLD_LINE in src:
    src = src.replace(OLD_LINE, NEW_LINE)
    fixes += 1
    print("Fix 1 applied: f-string format spec crash repaired")

# Generic scan: flag any other :.Nf if ... else patterns we missed
bad_fstr = re.compile(r'\{[^}]*?:\.[0-9]+f\s+if\s+[^}]*?\}')
remaining = bad_fstr.findall(src)
if remaining:
    print(f"WARNING: {len(remaining)} other invalid f-string pattern(s) found:")
    for m in remaining:
        print(f"  {m[:80]}")

# ------------------------------------------------------------------
# FIX 2: Add _fmt_score() helper so ALL score displays are safe
# ------------------------------------------------------------------
HELPER = '''
def _fmt_score(val) -> str:
    """Safely format a score float; returns - for NaN/None."""
    try:
        import pandas as pd
        if val is None or pd.isna(float(val)):
            return "-"
        return f"{float(val):.0f}"
    except Exception:
        return "-"

'''

if "_fmt_score" not in src:
    # Insert after the last top-level import line
    lines = src.split("\n")
    last_import_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = idx
    insert_pos = sum(len(l) + 1 for l in lines[:last_import_idx + 1])
    src = src[:insert_pos] + HELPER + src[insert_pos:]
    fixes += 1
    print("Fix 2 applied: _fmt_score() helper added")

# ------------------------------------------------------------------
# FIX 3: Replace remaining score:.0f patterns that could hit NaN
# ------------------------------------------------------------------
# Replace f"{score:.0f}" with _fmt_score(score) everywhere safe
score_fmt = re.compile(r'f"([^"]*)\{score:\.0f\}([^"]*)"')
count = len(score_fmt.findall(src))
if count:
    src = score_fmt.sub(lambda m: f'f"{m.group(1)}{{_fmt_score(score)}}{m.group(2)}"', src)
    fixes += count
    print(f"Fix 3 applied: {count} inline score:.0f replaced with _fmt_score()")

# ------------------------------------------------------------------
# Write result
# ------------------------------------------------------------------
if fixes:
    app_path.write_text(src, encoding="utf-8")
    print(f"\n{fixes} fix(es) applied to app.py")
    print("Restart Streamlit:  Ctrl+C  then  streamlit run app.py")
    print("Restore backup:     copy app.py.bak app.py")
else:
    print("No changes needed in app.py.")
    Path("app.py.bak").unlink(missing_ok=True)
