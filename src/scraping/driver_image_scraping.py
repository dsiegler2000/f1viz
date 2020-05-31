import requests
from bs4 import BeautifulSoup
from data_loading.data_loader import load_drivers
import pandas as pd

from utils import get_driver_name


def get_driver_image(wiki_url, surname):
    """
    Very fragile scraping method
    """
    html = requests.get(wiki_url)
    b = BeautifulSoup(html.text, "lxml")
    imgs = b.find_all(name="img")
    for img_tag in imgs:
        if "src" in img_tag.attrs:
            if surname not in img_tag.attrs["src"]:
                continue
            return img_tag.attrs["src"]
    return ""


drivers = load_drivers()
driver_img_src = pd.DataFrame(columns=["driverId", "imgUrl"])
i = 0
for did, row in drivers.iterrows():
    url = get_driver_image(row["url"], row["surname"]).strip("//")
    driver_img_src = driver_img_src.append({
        "imgUrl": url,
        "driverId": did
    }, ignore_index=True)
    i += 1
    name = get_driver_name(did, include_flag=False)
    print(f"{name}: {url}")
    print(f"{i} / {drivers.shape[0]}")
    print("=" * 20)
driver_img_src = driver_img_src.set_index("driverId")
driver_img_src.to_csv("data/static_data/driver_image_urls.csv")
