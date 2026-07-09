#!/usr/bin/env python3
"""
Mirror the Zenodo dataset record to a HuggingFace dataset repo.

Downloads the 8 .h5 files from Zenodo record 17399948
("Datasets for Self-Driving Trigger Study at L1", DOI 10.5281/zenodo.17399948)
into a staging folder, writes a dataset card, then uploads to HF.

Prereqs:
    pip install "huggingface_hub[hf_xet]"
    hf auth login --token hf_XXXX          # write-scoped token

Usage:
    python upload_zenodo_to_hf.py <hf_username>/<dataset_name> [--private]

Example:
    python upload_zenodo_to_hf.py zixinding/CMS-trigger-l1
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

ZENODO_RECORD = "17399948"
STAGING = Path("hf_dataset_upload")


def zenodo_files(record: str):
    url = f"https://zenodo.org/api/records/{record}"
    d = json.load(urllib.request.urlopen(url, timeout=60))
    return d["metadata"]["title"], d.get("doi"), d["files"]


def download(files):
    STAGING.mkdir(exist_ok=True)
    for f in sorted(files, key=lambda x: x["key"]):
        key, size = f["key"], f["size"]
        link = f["links"]["self"]
        dest = STAGING / key
        if dest.exists() and dest.stat().st_size == size:
            print(f"  ✓ {key} already staged ({size/1e6:.1f} MB)")
            continue
        print(f"  ↓ {key} ({size/1e6:.1f} MB) ...", flush=True)
        urllib.request.urlretrieve(link, dest)
        got = dest.stat().st_size
        if got != size:
            sys.exit(f"SIZE MISMATCH {key}: got {got} expected {size}")
    print("  All files staged.")


def write_card(title, doi, files, repo_id):
    rows = "\n".join(
        f"| `{f['key']}` | {f['size']/1e6:.1f} MB |"
        for f in sorted(files, key=lambda x: x["key"])
    )
    card = f"""---
license: cc-by-4.0
pretty_name: {title}
tags:
  - physics
  - high-energy-physics
  - anomaly-detection
  - reinforcement-learning
  - CMS-open-data
size_categories:
  - 1B<n<10B
---

# {title}

Mirror of Zenodo record [{doi}](https://doi.org/{doi}) — datasets for
*Learning to Trigger: Reinforcement Learning at the Large Hadron Collider*
([arXiv:2606.23993](https://arxiv.org/abs/2606.23993)).

Derived from **CMS 2016 Open Data** for Level-1 (L1) hadronic objects (jets).
Each file contains reconstructed jet features and the number of primary
vertices (N_PV) per event.

## Files

| File | Size |
|------|------|
{rows}

- **`MinBias_1.h5`** — min-bias MC background (AD autoencoder training).
- **`MinBias_2.h5`** — alternate min-bias MC background (control-algorithm studies).
- **`TT_1.h5`** — Standard Model t-tbar hadronic signal.
- **`HToAATo4B.h5`** — BSM H→AA→4b signal.
- **`data_Run_2016_283876.h5`** — real CMS 2016 run (AD training, real background).
- **`data_Run_2016_283408_longest.h5`** — longest CMS 2016 run (control-algorithm testing).
- **`Trigger_food_MC.h5`** — precomputed control variables (anomaly score, HT, N_PV) for MC.
- **`Trigger_food_Data.h5`** — precomputed control variables for real data (matched by N_PV).

## Loading

```python
import h5py
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="{repo_id}", filename="Trigger_food_MC.h5",
                       repo_type="dataset")
with h5py.File(path, "r") as f:
    print(list(f.keys()))
```

## Citation

```bibtex
@misc{{ding2026learning,
  title         = {{Learning to Trigger: Reinforcement Learning at the Large Hadron Collider}},
  author        = {{Ding, Zixin and Emami, Shaghayegh and Salvi, Giovanna and Tosciri, Cecilia and Gandrakota, Abhijith and Ngadiuba, Jennifer and Tran, Nhan and Herwig, Christian and Miller, David W. and Chen, Yuxin}},
  year          = {{2026}},
  eprint        = {{2606.23993}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.LG}},
  url           = {{https://arxiv.org/abs/2606.23993}}
}}
```
"""
    (STAGING / "README.md").write_text(card)
    print("  Wrote dataset card (README.md).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", nargs="?", help="HF dataset repo id, e.g. zixinding/CMS-trigger-l1")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--stage-only", action="store_true", help="download from Zenodo and exit (no auth/upload)")
    args = ap.parse_args()

    title, doi, files = zenodo_files(ZENODO_RECORD)
    print(f"Zenodo: {title}  ({doi})  — {len(files)} files")

    print("Staging files locally ...")
    download(files)

    if args.stage_only:
        print("Stage-only: done. Files in", STAGING.resolve())
        return
    if not args.repo_id:
        sys.exit("repo_id required unless --stage-only")

    write_card(title, doi, files, args.repo_id)

    api = HfApi()
    print(f"Creating dataset repo {args.repo_id} (private={args.private}) ...")
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    print("Uploading folder to HF ...")
    api.upload_folder(folder_path=str(STAGING), repo_id=args.repo_id,
                      repo_type="dataset",
                      commit_message=f"Mirror Zenodo {doi}")
    print(f"\nDone → https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
