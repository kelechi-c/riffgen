import os
import requests
import shutil
from selenium import webdriver
from selenium.webdriver.firefox.options import Options


def init_driver(link: str):
    # initialize driver
    drive_opts = Options()
    drive_opts.add_argument("-headless")
    driver = webdriver.Firefox(options=drive_opts)
    print("Driver init")

    # load page content
    driver.get(link)

    return driver


def audiofile_name(link: str):
    fn = os.path.basename(link)
    file_name = fn.lower()

    return file_name


def download_element(file_link, k, folder):
    response = requests.get(file_link, stream=True)
    title = audiofile_name(file_link)
    file_path = f"{title}_{k}.wav"
    file_path = os.path.join(folder, file_path)

    with open(file_path, "wb") as file:
        file.write(response.content)
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, file)
