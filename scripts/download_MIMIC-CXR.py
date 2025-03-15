import os
import requests
import shutil
import zipfile
import gzip
import math

# For HTML parsing
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# For concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed

# For DICOM -> JPG
import pydicom
import numpy as np
import cv2


# ------------------------------------------------
# 1. USER PROMPT FOR DEFAULT OR CUSTOM SETTINGS
# ------------------------------------------------
user_input = None
while user_input not in ["Y", "N"]:
    user_input = input("Default settings? No: N, Yes: Y ").strip().upper()

if user_input == "N":
    sessionID = input("Please Enter sessionid: ").strip()
    SAVE_DIR = input("Please Enter SAVE_DIR: ").strip()
    MAX_FILES = int(input("Please Enter MAX_FILES (max DICOM to convert): ").strip())

    # IMPORTANT: Use the page under 'content' for top-level files
    TOP_LEVEL_URL = "https://physionet.org/content/mimic-cxr/2.1.0/"

    # The subfolder containing raw DICOMs
    BASE_DICOM_URL = "https://physionet.org/files/mimic-cxr/2.1.0/files/"

    PN_COOKIES = {"sessionid": sessionID}
else:
    # Defaults
    PN_COOKIES = {'sessionid': '53moqg0un3pjfkilop0100x2xamn25jr'}  # example
    SAVE_DIR = "/content/Dataset"
    MAX_FILES = 8000

    # Notice: top-level is 'content/mimic-cxr/2.1.0/'
    TOP_LEVEL_URL = "https://physionet.org/content/mimic-cxr/2.1.0/"
    BASE_DICOM_URL = "https://physionet.org/files/mimic-cxr/2.1.0/files/"


# ------------------------------------------------
# 2. CREATE SESSION & TEST LOGIN
# ------------------------------------------------
session = requests.Session()
session.cookies.update(PN_COOKIES)

# Just test we can open the DICOM subfolder
test_resp = session.get(BASE_DICOM_URL)
if test_resp.status_code == 200:
    print("✅ Successful Login.")
else:
    print("❌ Login Failed or 403 Forbidden. Check your session cookie/credentials!")
    exit(1)


# ------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------

def human_readable_size(num_bytes):
    """Convert bytes to human-readable form."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes/1024:.1f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes/(1024**2):.1f} MB"
    else:
        return f"{num_bytes/(1024**3):.1f} GB"


def get_soup(url, s):
    """Return a BeautifulSoup object for the HTML at `url` (authenticated)."""
    r = s.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, 'html.parser')


def extract_if_archive(filepath):
    """
    Extract .zip or .gz in-place, then remove original.
    """
    if not os.path.isfile(filepath):
        return
    if filepath.lower().endswith('.zip'):
        folder = os.path.dirname(filepath)
        print(f"Extracting ZIP -> {folder}")
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                zf.extractall(folder)
            os.remove(filepath)
            print(f"Removed ZIP: {filepath}")
        except Exception as e:
            print(f"Error extracting {filepath}: {e}")

    elif filepath.lower().endswith('.gz'):
        outpath = filepath[:-3]  # remove .gz
        print(f"Extracting GZ -> {outpath}")
        try:
            with gzip.open(filepath, 'rb') as f_in, open(outpath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(filepath)
            print(f"Removed GZ: {filepath}")
        except Exception as e:
            print(f"Error extracting {filepath}: {e}")


def download_file(file_url, local_path, session):
    """
    Download a single file from file_url -> local_path.
    Returns (local_path, bytes_downloaded).
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with session.get(file_url, stream=True) as r:
        r.raise_for_status()
        size = 0
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
    return (local_path, size)


def dicom_to_jpg(dcm_path, jpg_path, target_size=1_000_000):
    """
    Convert a single DICOM file to ~1 MB JPG.
    Rescales pixel_array to [0..255], then tries multiple JPEG qualities.
    Removes original .dcm on success.
    """
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array
    except Exception as e:
        print(f"❌ Could not read DICOM: {dcm_path} => {e}")
        return

    arr = arr.astype(np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < 1e-6:
        arr[:] = 0
    else:
        arr = (arr - min_val)/(max_val - min_val)*255.0
    arr = arr.astype(np.uint8)

    quality = 90
    encoded = None
    while quality >= 10:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, encimg = cv2.imencode('.jpg', arr, encode_params)
        if not ok:
            print(f"❌ JPEG encode failed: {dcm_path}")
            return
        if len(encimg) <= target_size:
            encoded = encimg
            break
        quality -= 10
    if encoded is None:
        encoded = encimg

    # Save
    os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
    with open(jpg_path, 'wb') as f:
        f.write(encoded.tobytes())
    # Remove original .dcm
    try:
        os.remove(dcm_path)
    except:
        pass
    print(f"✔ DICOM -> JPG: {dcm_path} => {jpg_path}, size ~ {human_readable_size(len(encoded))}")


def relative_local_path(full_url):
    """
    For https://physionet.org/files/mimic-cxr/2.1.0/files/p10/p10000032/...,
    replicate 'files/p10/p10000032/...'.
    """
    split_key = "mimic-cxr/2.1.0/"
    parts = full_url.split(split_key, 1)
    if len(parts) < 2:
        return os.path.basename(full_url)
    else:
        return parts[1].lstrip("/")


# ------------------------------------------------
# 4. DOWNLOAD TOP-LEVEL FILES (PARALLEL)
# ------------------------------------------------
def download_top_level_files(session, top_level_url, save_dir):
    """
    Finds <a class="download"> from the page
      https://physionet.org/content/mimic-cxr/2.1.0/
    Then downloads them.
    """
    soup = get_soup(top_level_url, session)

    # 'download' anchors are the actual ?download links.
    download_anchors = soup.find_all('a', class_='download')

    exts = ('.zip', '.txt', '.gz')
    tasks = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for anchor in download_anchors:
            href = anchor.get('href', '')
            if not href:
                continue
            file_url = urljoin(top_level_url, href)
            parsed = urlparse(file_url)
            basename = os.path.basename(parsed.path)

            # If it ends with .zip, .txt, .gz
            # LICENSE.txt, SHA256SUMS.txt, .csv.gz, etc.
            if not basename.lower().endswith(exts):
                continue

            local_path = os.path.join(save_dir, basename)
            fut = executor.submit(download_file, file_url, local_path, session)
            tasks.append(fut)

        for fut in as_completed(tasks):
            try:
                dl_path, dl_size = fut.result()
                print(f"Downloaded {dl_path} ({human_readable_size(dl_size)})")
                extract_if_archive(dl_path)
            except Exception as e:
                print(f"Download failed: {e}")


# ------------------------------------------------
# 5. RECURSIVELY CRAWL .dcm FILES & CONVERT TO JPG
# ------------------------------------------------
def crawl_and_convert_dicom(session, start_url, save_dir, max_files):
    """
    Recursively crawls subfolders from `start_url` searching for .dcm files,
    downloads them, converts to ~1MB JPG, removes .dcm.
    """
    dcm_count = 0
    links_to_visit = [start_url]
    visited = set()

    while links_to_visit and dcm_count < max_files:
        current_url = links_to_visit.pop()
        if current_url in visited:
            continue
        visited.add(current_url)

        soup = get_soup(current_url, session)
        tasks = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if not href or href == '../':
                    continue
                full_url = urljoin(current_url, href)

                if href.endswith('/'):
                    # subdirectory
                    links_to_visit.append(full_url)
                elif href.lower().endswith('.dcm'):
                    # DICOM => schedule download & convert
                    rel_path = relative_local_path(full_url)
                    dcm_local_path = os.path.join(save_dir, rel_path)
                    jpg_local_path = dcm_local_path[:-4] + '.jpg'

                    fut = executor.submit(download_and_convert, full_url, dcm_local_path, jpg_local_path, session)
                    tasks.append(fut)

            for fut in as_completed(tasks):
                if dcm_count >= max_files:
                    break
                try:
                    success = fut.result()
                    if success:
                        dcm_count += 1
                        print(f"Processed {dcm_count} DICOMs so far.")
                    if dcm_count >= max_files:
                        break
                except Exception as e:
                    print(f"Error on DICOM: {e}")

        if dcm_count >= max_files:
            break


def download_and_convert(file_url, dcm_path, jpg_path, session):
    """
    1) Download DICOM
    2) Convert -> 1MB JPG
    """
    try:
        _, size_dl = download_file(file_url, dcm_path, session)
        print(f"Downloaded DICOM: {dcm_path} ({human_readable_size(size_dl)})")
    except Exception as e:
        print(f"Failed to download {file_url}: {e}")
        return False

    try:
        dicom_to_jpg(dcm_path, jpg_path, target_size=1_000_000)
    except Exception as ex:
        print(f"Failed to convert {dcm_path} -> JPG: {ex}")
        return False

    return True


# ------------------------------------------------
# 6. MAIN EXECUTION
# ------------------------------------------------
# A) Download top-level files (like LICENSE.txt, csv.gz, etc.) from content page
download_top_level_files(session, TOP_LEVEL_URL, SAVE_DIR)

## Alternative way to download top level files
#!wget -r -N -c -np --user nexes2024 --ask-password https://physionet.org/files/mimic-cxr/2.1.0

# B) Crawl the "files/" subdir, find .dcm, convert to JPG
crawl_and_convert_dicom(session, BASE_DICOM_URL, SAVE_DIR, MAX_FILES)

print("✅ All done.")
