"""
Wrapped script for running improved training from `scripts/`.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_improved import main

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=False)
    args = parser.parse_args()
    main(args)
