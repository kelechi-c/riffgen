import os
import selenium
from selenium.webdriver.common.by import By
from scraper_utils import download_element, audiofile_name, init_driver
from .config import scraper_config
from tqdm.auto import tqdm


class MusicScraper:
    def __init__(self, config=scraper_config):
        self.site_link = config.freesound_url_for_music
        self.driver = init_driver(self.site_link)
        self.config = config

    def link_collector(self):
        section_links = self.driver.find_elements(
            By.CSS_SELECTOR, self.config.player_id
        )

        audio_links = [
            x.get_attribute(self.config.mp3_attr) for x in tqdm(section_links)
        ]

        print(f"got {len(audio_links)} links")
        return audio_links

    def download_tracks(self, links):
        k = 0
        folder = self.config.outpath

        if os.path.exists(folder) is not True:
            os.mkdir(folder)

        for link in tqdm(links):
            download_element(link, k, folder)
