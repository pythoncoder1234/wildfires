from time import time

import bs4
import requests

from variables import *


def download_all(url, folder, depth=1):
    if not url.endswith("/"):
        url += "/"

    print("Fetching", url)
    req = requests.get(url)
    soup = bs4.BeautifulSoup(req.content, "html.parser")
    links = soup.select("a")

    if not len(links):
        print("Invalid link")
        return

    for i, link in enumerate(links):
        if link.text == "Parent Directory":
            continue

        if "varges" in link.text:
            continue

        if depth > 1:
            folder_path = folder + "/" + link.text
            os.mkdir(folder_path)
            download_all(url + link["href"], folder_path, depth - 1)

        else:
            download_file(link.text, url + link["href"], folder)

    print("Done!")


def download_file(name, link, folder):
    try:
        with open(f"{folder}/{name}", "xb") as out:
            start = time()
            req = requests.get(link)
            print(f"Downloaded {name} in {round(time() - start, 1)}s")

            out.write(req.content)

    except FileExistsError:
        try:
            with open(f"{folder}/{name}", "rb") as existing_file:
                next(existing_file)
            print(f"File {name} exists, skipping download")

        except StopIteration:
            os.remove(f"{folder}/{name}")
            download_file(name, link, folder)


print(os.getcwd())
download_all(URL, DATASET_FOLDER)
