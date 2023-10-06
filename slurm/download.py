import json
import urllib.request
import sys
import time


def download(url, path, retries=3):
    for _ in range(retries + 1):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f'Downloading failed with {e}')
            time.sleep(60)
        else:
            return


if __name__ == '__main__':
    params = json.load(sys.stdin)
    urls = params['gtfs_url']
    paths = params['gtfs_path']

    for url, path in zip(urls, paths):
        download(url, path)
