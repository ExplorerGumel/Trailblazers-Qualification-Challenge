"""
Wrapper to run `run_quick.py` from project root using scripts folder.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_quick import args, main as _main

if __name__ == '__main__':
    _main(args)
