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

    # load page content
    driver.get(link)

    return driver


def audiofile_name(link: str, folder: str):
    fn = os.path.basename(link)
    file_name = fn.lower()
    file_path = os.path.join(folder, file_name)

    return file_path


def download_element(file_link, folder):
    response = requests.get(file_link, stream=True)
    file_path = audiofile_name(file_link, folder)

    with open(file_path, "wb") as file:
        file.write(response.content)
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, file)
