"""Download NASA CMAPSS FD001 dataset via Kaggle API."""
import subprocess
import sys
from pathlib import Path

DATASET = "behrad3d/nasa-cmapss"
OUTPUT_DIR = Path("data/raw")


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} to {OUTPUT_DIR}...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET,
         "-p", str(OUTPUT_DIR), "--unzip"],
        check=True
    )
    print("Download complete. Files in data/raw/:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  {f.name}")


if __name__ == "__main__":
    download()
