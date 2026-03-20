"""Download / verify face models into DATA_DIR (default: ~/.omnivision/data)."""
import os
from pathlib import Path

# Override with env DATA_ROOT if needed
DATA_DIR = os.environ.get("DATA_ROOT", str(Path.home() / ".omnivision" / "data"))
