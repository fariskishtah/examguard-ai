"""Freeze the dev/test split for the REAL-image phone benchmark (COCO val2017, mirror on Hugging Face).

COCO val2017 images were NOT used to train the Ultralytics weights (trained on train2017), so they are a
genuine held-out set for the generic detector. They are NOT exam scenes. Images are used locally only
(mixed Flickr licences; annotations CC BY 4.0) and are never redistributed - only image ids are stored.
Rule: split by md5(image_id) parity -> dev (all tuning/selection) vs test (reported once at the end).
"""
import glob
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PHONE, PERSON = 67, 0


def main():
    pos, neg = [], []
    for f in sorted(glob.glob(str(ROOT / "data/eval/coco_hf/*.parquet"))):
        p = pq.ParquetFile(f)
        for rg in range(p.num_row_groups):
            for r in p.read_row_group(rg, columns=["image_id", "objects"]).to_pylist():
                cats = r["objects"]["category"]
                if PHONE in cats:
                    pos.append(r["image_id"])
                elif PERSON in cats:
                    neg.append(r["image_id"])
    rng = random.Random(0)
    neg = sorted(rng.sample(sorted(neg), 400))
    side = lambda i: "dev" if int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 2 == 0 else "test"  # noqa: E731
    out = {"source": "COCO val2017 via Hugging Face detection-datasets/coco (local eval only)", "seed": 0,
           "rule": "md5(image_id) parity: even=dev, odd=test; negatives = 400 random val images with a person and NO phone",
           "dev": {"positives": sorted(i for i in pos if side(i) == "dev"), "negatives": [i for i in neg if side(i) == "dev"]},
           "test": {"positives": sorted(i for i in pos if side(i) == "test"), "negatives": [i for i in neg if side(i) == "test"]}}
    (ROOT / "artifacts/metrics/phone_coco_split.json").write_text(json.dumps(out))
    print({k: {kk: len(vv) for kk, vv in v.items()} for k, v in out.items() if k in ("dev", "test")})


if __name__ == "__main__":
    main()
