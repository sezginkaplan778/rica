import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os, re, time

base_url = "https://www.richmondcollege.co.uk"
#base_url = "https://www.europarl.europa.eu/committees/en/documents"

visited = set()
downloaded = set()

headers = {"User-Agent": "Mozilla/5.0 (compatible; FileCrawler/1.0)"}

def crawl(url, depth=0, max_depth=3):
    if depth > max_depth or url in visited:
        return
    visited.add(url)

    try:
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # PDF, PNG, JPG, DOCX bağlantılarını bul
        for link in soup.find_all("a", href=True):
            href = link['href']
            if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            if any(href.lower().endswith(ext) for ext in [".pdf", ".png", ".jpg", ".jpeg", ".docx"]):
                file_url = urljoin(base_url, href)
                download_file(file_url)

        # <img src="..."> görsellerini indir
        for img in soup.find_all("img", src=True):
            src = img['src']
            #if any(src.lower().endswith(ext) for ext in [".pdf", ".docx"]):
            if any(ext in src.lower() for ext in [".pdf", ".png", ".jpg", ".jpeg", ".docx"]):
                # indir

                file_url = urljoin(base_url, src)
                download_file(file_url)

        # Alt sayfaları gez
        for link in soup.find_all("a", href=True):
            next_url = urljoin(base_url, link['href'])
            if base_url.split("//")[1].split("/")[0] in next_url:
                crawl(next_url, depth + 1, max_depth)

        time.sleep(0.5)

    except Exception as e:
        print(f"Error crawling {url}: {e}")

def download_file(file_url):
    if file_url in downloaded:
        return
    downloaded.add(file_url)
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", file_url.split("/")[-1])
    filepath = os.path.join("downloads", filename)

    print(f"Downloading: {file_url}")
    try:
        response = requests.get(file_url, headers=headers)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Saved to: {filepath}")
        else:
            print(f"Failed to download {file_url} (status {response.status_code})")
    except Exception as e:
        print(f"Error downloading {file_url}: {e}")

os.makedirs("downloads", exist_ok=True)
crawl(base_url, max_depth=3)

print("Ziyaret edilen sayfa sayısı:", len(visited))
print("İndirilen dosya sayısı:", len(downloaded))

