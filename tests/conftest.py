import sys
from pathlib import Path

# Test this checkout's source even when another checkout's pianopipe is installed (pip install -e elsewhere).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
