"""Download openly licensed sample videos from Wikimedia Commons and record provenance.

Files are saved under data/samples/ (git-ignored: we do not redistribute them).
Provenance (license, author, source URL, sha256) goes to data/samples/MANIFEST.json,
which IS committed so results are reproducible and attribution is preserved.
Usage: python scripts/fetch_samples.py [--list]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples"
UA = {"User-Agent": "examguard-ai/0.1 (open-source research; https://github.com/fariskishtah/examguard-ai)"}

# Exact Commons titles. Chosen only if the licence permits use; see docs/DATA_SOURCES.md.
TITLES = [
    "File:Video Codec Test pedestrian area 1080p25.y4m.webm",   # CC0
    "File:Student studies in classroom during face to face learning in Padang, Indonesia.webm",  # CC BY-SA 4.0
    "File:Black smartphone in hand (Unsplash).jpg",  # CC0, phone photo used ONLY for synthetic test fixtures
    "File:Smartphone in Hand (Unsplash).jpg",  # CC0, extra phone photos for HELD-OUT synthetic fixtures
    "File:Hand-smartphone-technology-calling.jpg",  # CC0
    "File:Smartphone Use.jpg",  # CC0
]


def api(params: dict) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def info(title: str) -> dict:
    d = api({"action": "query", "titles": title, "prop": "imageinfo",
             "iiprop": "url|size|mime|extmetadata|sha1",
             "iiextmetadatafilter": "LicenseShortName|Artist|LicenseUrl|Credit"})
    page = next(iter(d["query"]["pages"].values()))
    if "imageinfo" not in page:
        raise SystemExit(f"not found on Commons: {title}")
    ii = page["imageinfo"][0]
    md = ii.get("extmetadata", {})
    strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()  # noqa: E731
    return {"title": title, "url": ii["url"], "page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            "license": md.get("LicenseShortName", {}).get("value", "UNKNOWN"),
            "license_url": md.get("LicenseUrl", {}).get("value"), "author": strip(md.get("Artist", {}).get("value")),
            "width": ii["width"], "height": ii["height"], "bytes": ii["size"], "mime": ii["mime"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mf = OUT / "MANIFEST.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    for t in TITLES:
        meta = info(t)
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", t.removeprefix("File:"))
        dest = OUT / name
        if "--list" in sys.argv:
            print(meta["license"], meta["width"], meta["height"], meta["bytes"] // 1024, "KB", t)
            continue
        if not dest.exists():
            print(f"downloading {name} ({meta['bytes'] // 1024} KB, {meta['license']}) ...")
            req = urllib.request.Request(meta["url"], headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        meta["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
        meta["file"] = name
        manifest[name] = meta
    if "--list" not in sys.argv:
        mf.write_text(json.dumps(manifest, indent=2))
        print("manifest updated:", mf)


if __name__ == "__main__":
    main()
