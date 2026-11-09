"""Stage validate: run validate.py, save its JSON as reports/validation_report.json,
and exit non-zero (failing the pipeline) if validation did not pass."""
import json
import subprocess
import sys
from pathlib import Path

res = subprocess.run([sys.executable, "src/data/validate.py"], capture_output=True, text=True)
Path("reports").mkdir(exist_ok=True)
Path("reports/validation_report.json").write_text(res.stdout)
print(res.stdout)
if res.stderr:
    print(res.stderr, file=sys.stderr)
try:
    passed = bool(json.loads(res.stdout)["passed"])
except (ValueError, KeyError):
    passed = False
sys.exit(0 if (res.returncode == 0 and passed) else 1)