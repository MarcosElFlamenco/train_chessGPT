import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile
from tqdm import tqdm
import time

def fetch_twic_pgn_zip_links(session, base_url):
    """
    Fetch all ZIP file links associated with 'PGN' buttons from the TWIC base URL.
    """
    try:
        response = session.get(base_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching TWIC page: {e}")
        # Optional: Print more details for debugging
        # print(f"Response Status Code: {response.status_code}")
        # print(f"Response Headers: {response.headers}")
        # print(f"Response Content: {response.text[:500]}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    zip_links = []

    # Find all <a> tags with text containing 'PGN' (case-insensitive)
    a_tags = soup.find_all('a', string=lambda text: text and 'PGN' in text.upper())
    for a_tag in a_tags:
        href = a_tag.get('href', '')
        if href.lower().endswith('.zip'):
            full_url = urljoin(base_url, href)
            zip_links.append(full_url)

    return zip_links

def download_file(session, url, dest_path):
    """
    Download a file from a URL to the destination path with a progress bar.
    """
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            tqdm_desc = os.path.basename(dest_path)
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=tqdm_desc) as t:
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            t.update(len(chunk))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    return True

def unzip_file(zip_path, extract_to):
    """
    Unzip a ZIP file to the specified directory.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except zipfile.BadZipFile as e:
        print(f"Error unzipping {zip_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error unzipping {zip_path}: {e}")
        return False

def main():
    # Configuration
    base_url = 'https://theweekinchess.com/twic'
    download_dir = 'twic_zips_pgn'
    extract_dir = 'twic_extracted_pgn'

    # Create directories if they don't exist
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    # Initialize a session with headers to mimic a real browser
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' \
                      'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                      'Chrome/112.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': base_url,
        'Connection': 'keep-alive',
    })

    # Optional: Implement retries for transient errors
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Wait time between retries: {backoff factor} * (2 ** (retry number))
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    print("Fetching 'PGN' ZIP links from TWIC...")
    zip_links = fetch_twic_pgn_zip_links(session, base_url)
    if not zip_links:
        print("No 'PGN' ZIP files found.")
        return

    print(f"Found {len(zip_links)} 'PGN' ZIP file(s).")

    for zip_url in zip_links:
        zip_filename = os.path.basename(zip_url)
        download_path = os.path.join(download_dir, zip_filename)

        # Download the ZIP file if it doesn't already exist
        if not os.path.isfile(download_path):
            print(f"Downloading {zip_filename}...")
            success = download_file(session, zip_url, download_path)
            if not success:
                print(f"Failed to download {zip_url}. Skipping.")
                continue
            # Politeness delay to mimic human browsing
            time.sleep(2)
        else:
            print(f"{zip_filename} already exists. Skipping download.")

        # Unzip the file
        print(f"Unzipping {zip_filename}...")
        success = unzip_file(download_path, extract_dir)
        if not success:
            print(f"Failed to unzip {download_path}. Skipping.")
            continue

        # Optional: Delete the ZIP file after extraction
        # os.remove(download_path)

        # Additional politeness delay
        time.sleep(1)

    print("All 'PGN' ZIP files have been processed.")

if __name__ == "__main__":
    main()
