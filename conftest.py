"""Pytest configuration: add project root to sys.path so `src` is importable."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
