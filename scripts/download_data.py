#!/usr/bin/env python3
"""
Download Script — Radar Micro-Doppler Dataset
==============================================
Downloads the helicopter micro-Doppler dataset from HuggingFace Hub.

Dataset: https://huggingface.co/datasets/bukac82/MicroDopplerSignatures

Usage:
    # Recommended — uses HuggingFace Hub library (handles resuming, caching):
    python scripts/download_data.py

    # Download to a custom directory:
    python scripts/download_data.py --dest ./data

    # Generate locally instead of downloading (takes ~20-40 min):
    python scripts/download_data.py --source generate
"""

import argparse
import sys
from pathlib import Path

# ── HuggingFace dataset config ────────────────────────────────────────────────
HF_REPO_ID  = "bukac82/MicroDopplerSignatures"
HF_REPO_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

# Files to download from the dataset repo
HF_FILES = [
    "helicopter_microdoppler_dataset.csv",
    "helicopter_microdoppler_extended_dataset.csv",
]

# ── Local generation scripts (fallback) ──────────────────────────────────────
GENERATE_SCRIPT_BASE     = "Dataset/generate_dataset.py"
GENERATE_SCRIPT_EXTENDED = "Dataset/generate_dataset_extended.py"


def download_with_hf_hub(dest: Path):
    """Download using huggingface_hub — supports resume, progress bar, caching."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ✗ huggingface_hub not installed.")
        print("    Run: pip install huggingface-hub")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)

    for filename in HF_FILES:
        out_path = dest / filename
        if out_path.exists():
            print(f"  ⏭  {filename} already exists — skipping.")
            continue

        print(f"\n  ⬇  Downloading {filename} from HuggingFace...")
        print(f"     Source : {HF_REPO_URL}")
        print(f"     Dest   : {out_path}")

        try:
            cached_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(dest),
                local_dir_use_symlinks=False,  # copy actual file, not symlink
            )
            size_gb = Path(cached_path).stat().st_size / 1e9
            print(f"  ✔  Saved {filename} ({size_gb:.2f} GB)")
        except Exception as e:
            print(f"  ✗  Failed to download {filename}: {e}")
            print(f"\n  💡 Try manually:\n     pip install huggingface-hub")
            print(f"     huggingface-cli download {HF_REPO_ID} {filename} --repo-type dataset --local-dir .")
            sys.exit(1)


def download_with_wget_fallback(dest: Path):
    """Fallback: direct URL download via urllib (no resume support for large files)."""
    import urllib.request
    import shutil

    base_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/"
    dest.mkdir(parents=True, exist_ok=True)

    for filename in HF_FILES:
        out_path = dest / filename
        if out_path.exists():
            print(f"  ⏭  {filename} already exists — skipping.")
            continue
        url = base_url + filename
        print(f"\n  ⬇  Downloading {filename} ...")
        try:
            with urllib.request.urlopen(url) as response, open(out_path, "wb") as f:
                shutil.copyfileobj(response, f)
            size_gb = out_path.stat().st_size / 1e9
            print(f"  ✔  Saved {filename} ({size_gb:.2f} GB)")
        except Exception as e:
            print(f"  ✗  Failed: {e}")
            sys.exit(1)


def generate_locally(dest: Path):
    """Fallback: generate the datasets locally by running the generation scripts."""
    import subprocess

    print("\n📡 Generating datasets locally (this will take ~20–40 minutes)...")
    root = Path(__file__).parent.parent
    for script in [GENERATE_SCRIPT_BASE, GENERATE_SCRIPT_EXTENDED]:
        script_path = root / script
        if not script_path.exists():
            print(f"  ✗ Script not found: {script_path}")
            continue
        print(f"  Running {script} ...")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(dest),
        )
        if result.returncode != 0:
            print(f"  ✗ Script failed: {script}")
        else:
            print(f"  ✔ Done: {script}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the Radar Micro-Doppler dataset from HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Dataset URL:
  {HF_REPO_URL}

Examples:
  python scripts/download_data.py                     # download to project root
  python scripts/download_data.py --dest ./data       # download to ./data/
  python scripts/download_data.py --source generate   # generate locally instead
        """,
    )
    parser.add_argument(
        "--dest", default=".",
        help="Directory to save datasets (default: project root)"
    )
    parser.add_argument(
        "--source", default="huggingface",
        choices=["huggingface", "generate"],
        help="Data source (default: huggingface)"
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use direct URL download instead of huggingface_hub library"
    )
    args = parser.parse_args()

    dest = Path(args.dest).resolve()

    print(f"\n🚀 Radar Micro-Doppler Dataset Downloader")
    print(f"   Dataset : {HF_REPO_URL}")
    print(f"   Dest    : {dest}")
    print(f"   Source  : {args.source}\n")

    if args.source == "generate":
        generate_locally(dest)
    elif args.fallback:
        download_with_wget_fallback(dest)
    else:
        download_with_hf_hub(dest)

    print("\n✅ Dataset ready.")
    print(f"   Files saved to: {dest}")


if __name__ == "__main__":
    main()
