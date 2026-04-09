"""
app.py — Root-level Streamlit entry point.
Delegates to dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dashboard.app import main

if __name__ == "__main__":
    main()