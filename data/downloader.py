
import os
import requests
import zipfile
from tqdm import tqdm

"""
BASE_URL = "https://zenodo.org/records/4004271/files"
OUT_DIR = "kul_data"
os.makedirs(OUT_DIR, exist_ok=True)

def download(url, out_path, retries=5):
    temp_path = out_path + ".part"

    for attempt in range(retries):
        try:
            headers = {}
            downloaded = 0

            if os.path.exists(temp_path):
                downloaded = os.path.getsize(temp_path)
                headers["Range"] = f"bytes={downloaded}-"

            r = requests.get(url, stream=True, headers=headers, timeout=30)
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0)) + downloaded

            with open(temp_path, "ab") as f, tqdm(
                desc=os.path.basename(out_path),
                total=total,
                initial=downloaded,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            os.rename(temp_path, out_path)
            return

        except Exception as e:
            print(f"Retry {attempt + 1}/{retries} failed: {e}")

    raise RuntimeError(f"Failed to download {url}")

# ----------------------------
# THIS PART WAS MISSING 👇
# ----------------------------
if __name__ == "__main__":
    # Download S1–S16
    for i in range(1, 17):
        fname = f"S{i}.mat"
        download(f"{BASE_URL}/{fname}", f"{OUT_DIR}/{fname}")

    # Download stimuli
    zip_path = f"{OUT_DIR}/stimuli.zip"
    download(f"{BASE_URL}/stimuli.zip", zip_path)

    # Unzip
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(OUT_DIR)

    print("All downloads completed ✅")
"""
# DTU DATASET DOWNLOADER

