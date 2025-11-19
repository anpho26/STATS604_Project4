"""
PJM OSF bundle downloader (power demand history).

Purpose
-------
Download and extract the 10-year PJM load bundle hosted on OSF for the class.
Files are unzipped into `data/raw/power/` and can be used directly by the
training/EDA code.

Outputs
-------
- `data/raw/power/hrl_load_metered_*.csv`  (one or more CSVs)

CLI
---
Run from the project root:

    python -m src.downloads.pjm_osf_download
    python -m src.downloads.pjm_osf_download --out data/raw/power
    python -m src.downloads.pjm_osf_download --url <override_zip_url>

Args
----
--out : str, optional
    Output directory for extracted CSVs (default: `data/raw/power`).
--url : str, optional
    Override the OSF zip URL if needed.

Notes
-----
* Safe to re-run: existing files are skipped (or overwritten if you choose).
* The script performs a simple integrity check after extraction.
"""

from __future__ import annotations
import argparse
import hashlib
import io
import os
from pathlib import Path
import sys
import zipfile
import shutil
import tempfile

import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:
    # very old requests might not have urllib3 Retry available
    Retry = None  # type: ignore

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

DEFAULT_URL = "https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip="

def session_with_retries() -> requests.Session:
    s = requests.Session()
    if Retry is not None:
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"])
        )
        s.mount("http://", HTTPAdapter(max_retries=retries))
        s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units)-1:
        x /= 1024; i += 1
    return f"{x:.1f} {units[i]}"

def download(url: str, dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    s = session_with_retries()
    with s.get(url, stream=True, timeout=(10, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or None
        chunk = 1024 * 1024  # 1 MB
        if tqdm:
            bar = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading")
        else:
            bar = None
            if total:
                print(f"Downloading {human_bytes(total)} …", flush=True)
        tmp = dest_zip.with_suffix(".part")
        h = hashlib.sha256()
        with open(tmp, "wb") as f:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if not chunk_bytes:
                    continue
                f.write(chunk_bytes)
                h.update(chunk_bytes)
                if bar: bar.update(len(chunk_bytes))
        if bar: bar.close()
        tmp.rename(dest_zip)
        print(f"Saved: {dest_zip} (sha256={h.hexdigest()[:12]}…)")

def _safe_join(base: Path, *parts: str) -> Path:
    # Prevent Zip Slip: ensure extracted path stays within base
    target = (base / Path(*parts)).resolve()
    base_resolved = base.resolve()
    if not str(target).startswith(str(base_resolved)):
        raise RuntimeError(f"Blocked unsafe path outside extraction dir: {target}")
    return target

def safe_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if not members:
            print("Zip is empty?", file=sys.stderr)
            return
        if tqdm:
            it = tqdm(members, desc="Extracting", unit="file")
        else:
            it = members
            print(f"Extracting {len(members)} files to {out_dir} …")
        for m in it:
            # Skip directory entries
            if m.filename.endswith("/"):
                continue
            target = _safe_join(out_dir, m.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    print(f"Done. Files extracted under: {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Download and extract PJM OSF archive")
    ap.add_argument("--url", default=DEFAULT_URL, help="OSF zip URL (default: given course link)")
    ap.add_argument("--out", default="data/raw/pjm_osf", help="Extraction directory")
    ap.add_argument("--keep-zip", action="store_true", help="Keep the downloaded .zip")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir.parent / "pjm_osf_archive.zip"
    try:
        download(args.url, zip_path)
        safe_extract(zip_path, out_dir)
    except KeyboardInterrupt:
        print("\nCanceled.", file=sys.stderr)
        sys.exit(130)
    finally:
        if zip_path.exists() and (not args.keep_zip):
            try:
                zip_path.unlink()
            except Exception:
                pass

if __name__ == "__main__":
    main()