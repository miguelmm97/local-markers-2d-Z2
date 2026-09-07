import sys
from pathlib import Path

# Make the real "modules" package (at the repo root) importable from any test
# file, the same way base-code/*.py scripts reach it via the base-code/modules
# symlink -- here we just point sys.path at the repo root directly instead
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
