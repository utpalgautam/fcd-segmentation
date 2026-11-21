"""
Executable module entry point for fcd_segmentation.preprocessing.pipeline.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_dataset import main

if __name__ == "__main__":
    sys.exit(main())
