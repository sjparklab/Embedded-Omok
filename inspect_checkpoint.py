#!/usr/bin/env python3
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
args = parser.parse_args()
ck = torch.load(args.path, map_location='cpu')
print("Type:", type(ck))
if isinstance(ck, dict):
    print("Keys:", list(ck.keys()))
    # if 'model' key exists, inspect that
    if 'model' in ck:
        sd = ck['model']
    elif 'state_dict' in ck:
        sd = ck['state_dict']
    else:
        # maybe ck is state_dict already
        sd = ck
else:
    sd = ck
print("Number of params in state_dict:", len(sd))
for k in list(sd.keys())[:30]:
    v = sd[k]
    try:
        print(k, getattr(v, "shape", type(v)))
    except Exception:
        print(k, type(v))