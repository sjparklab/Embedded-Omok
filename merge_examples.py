#!/usr/bin/env python
# merge_examples.py
# Merge per-worker pickle files in a directory into a single examples.pkl
import argparse
import os
import pickle
import glob
import random

def load_worker_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="examples")
    parser.add_argument("--out_file", type=str, default="examples.pkl")
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    if not files:
        print("No example files found in", args.input_dir)
        return
    all_examples = []
    for p in files:
        try:
            ex = load_worker_file(p)
            if isinstance(ex, list):
                all_examples.extend(ex)
            else:
                print("Warning: unexpected content in", p)
        except Exception as e:
            print("Failed to load", p, e)
    if args.shuffle:
        random.shuffle(all_examples)
    with open(args.out_file, "wb") as f:
        pickle.dump(all_examples, f)
    print(f"Merged {len(all_examples)} examples into {args.out_file}")

if __name__ == "__main__":
    main()