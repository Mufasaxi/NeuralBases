import os
import requests
from bs4 import BeautifulSoup
import time

def download_file(url, dest_folder, retries=5, backoff_factor=0.5):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    local_filename = os.path.join(dest_folder, url.split('/')[-1])
    
    # Skip downloading if the file already exists
    if os.path.exists(local_filename):
        print(f"File {local_filename} already exists. Skipping.")
        return local_filename
    
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return local_filename
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                print(f"Failed to download {url} after {retries} attempts. Skipping.")
                return None

def get_file_urls(base_url):
    if not base_url.endswith('/'):
        base_url += '/'
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    return [base_url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.rtbw')]

sixman_base_url = "https://tablebase.lichess.ovh/tables/standard/6-wdl/"
threeFourFive_base_url = "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/"
dest_folder = "./endgame_tablebase"

try:
    file_urls = get_file_urls(sixman_base_url)
    if not file_urls:
        print("No files found to download.")
    else:
        for file_url in file_urls:
            print(f"Downloading {file_url}...")
            download_file(file_url, dest_folder)
        print("All files downloaded.")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

try:
    file_urls = get_file_urls(threeFourFive_base_url)
    if not file_urls:
        print("No files found to download.")
    else:
        for file_url in file_urls:
            print(f"Downloading {file_url}...")
            download_file(file_url, dest_folder)
        print("All files downloaded.")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
