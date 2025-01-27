import os
import re
import time
import argparse
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download specific links from a website based on a regex pattern.")
    wcex = r'worldchamp\d{4}.pgn'
    fcex = r'FideChamp\d{4}.pgn' 
    genex = r".*\d{4}.pgn"
    parser.add_argument('-r', '--regex', default=genex,
                        help='Regex pattern to match link labels. Default is "".')
    parser.add_argument('-s', '--sleep', type=float, default=1.0, 
                        help='Sleep time in seconds between downloads. Default is 1 second.')
    parser.add_argument('-o', '--output', default='downloads', 
                        help='Directory to save downloaded files. Default is "./downloads".')
    parser.add_argument('-u', '--url_pattern', default=None,
                        help='Optional additional URL pattern to filter links. If not provided, all matching links will be considered.')
    return parser.parse_args()

def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_links(html, base_url, pattern):
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        link_label = a_tag.get_text(strip=True)
        href = a_tag['href']
        if re.search(pattern, link_label, re.IGNORECASE):
            full_url = urljoin(base_url, href)
            links.add(full_url)
    return list(links)

def sanitize_filename(url):
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = 'index.html'
    return filename

def download_link(url, output_dir):
    try:
        response = requests.get(url)
        response.raise_for_status()
        filename = sanitize_filename(url)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return True
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False
    except OSError as e:
        print(f"Failed to save {url} to {file_path}: {e}")
        return False

def main():
    args = parse_arguments()
    regex_pattern = args.regex
    sleep_time = args.sleep
    output_dir = args.output
    base_url = "https://www.pgnmentor.com/files.html#players"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching HTML content from {base_url}...")
    html_content = fetch_html(base_url)
    if not html_content:
        print("Failed to retrieve the webpage. Exiting.")
        return

    print(f"Extracting links matching pattern: {regex_pattern}")
    matched_links = extract_links(html_content, base_url, regex_pattern)
    if not matched_links:
        print("No matching links found. Exiting.")
        return

    print(f"Found {len(matched_links)} matching links.")
    print(f"Starting download of {len(matched_links)} files...")

    for link in tqdm(matched_links, desc="Downloading", unit="file"):
        success = download_link(link, output_dir)
        if success:
            tqdm.write(f"Downloaded: {link}")
        else:
            tqdm.write(f"Failed: {link}")
        time.sleep(sleep_time)

    print("Download process completed.")

if __name__ == "__main__":
    main()
