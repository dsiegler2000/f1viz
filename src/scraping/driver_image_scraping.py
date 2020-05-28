import requests
from bs4 import BeautifulSoup
from data_loading.data_loader import load_drivers
import pandas as pd


def get_driver_image(wiki_url):
    """
    Very fragile scraping method
    """
    html = requests.get(wiki_url)
    b = BeautifulSoup(html.text, "lxml")
    imgs = b.find_all(name="img")
    if len(imgs) > 0:
        img_tag = imgs[0]
        if "src" in img_tag.attrs:
            return img_tag.attrs["src"]
    return ""


drivers = load_drivers()
driver_img_src = pd.DataFrame(columns=["driverId", "imgUrl"])
i = 0
for idx, row in drivers.iterrows():
    url = get_driver_image(row["url"]).strip("//")
    driver_img_src = driver_img_src.append({
        "url": url,
        "driverId": idx
    }, ignore_index=True)
    if "flag" in url.lower():
        driver_name = row["forename"] + " " + row["surname"]
        print(driver_name)
        print(row["url"])
        print(url)
    i += 1
    print(f"{i} / {drivers.shape[0]}")
    print("=" * 20)
driver_img_src = driver_img_src.set_index("driverId")
driver_img_src.to_csv("data/static_data/driver_image_urls.csv")
