import os
from selenium.webdriver.common.by import By
from scraper_utils import download_element, init_driver, audiofile_name
from .config import scraper_config
from tqdm.auto import tqdm
import pandas as pd


class FreeSoundMusicScraper:
    def __init__(self, config=scraper_config):
        self.config = config
        self.page_links = [
            f"https://freesound.org/browse/tags/?f=tag%3A%22music%22&page={page_num}#sound"
            for page_num in range(1, self.config.num_pages)
        ]

    def page_crawler(self):
        # for each page, will return metadata which includes
        # link[url], title[str], duration[float], tags[list]
        link_holder = []
        durations = []
        tag_list = []
        title_list = []
        file_list = []

        for pg_link in tqdm(self.page_links):
            print(f"crawling current page => {pg_link}")
            page_driver = init_driver(pg_link)

            scraped_file_data = self._get_links(page_driver)
            titles = self._get_title(page_driver)
            # tags = self._get_tags(page_driver)

            link_holder.extend(scraped_file_data["audio_links"])
            durations.extend(scraped_file_data["duration"])
            file_list.extend(scraped_file_data["file_name"])

            # tag_list.extend(tags)
            title_list.extend(titles)

            page_driver.quit()

        print(f"total links {len(link_holder)}")

        return {
            "file_name": file_list,
            "links": link_holder,
            "title": title_list,
            "durations": durations,
            #             "tags": tag_list
        }

    def _get_links(self, driver):
        section_links = driver.find_elements(
            By.CSS_SELECTOR, self.config.player_id)

        audio_links = [
            x.get_attribute(self.config.mp3_attr)
            for x in tqdm(section_links, desc="retrieving links")
        ]

        file_paths = [audiofile_name(x, self.config.outpath)
                      for x in audio_links]

        durations = [
            float(x.get_attribute(self.config.duration_attr)) for x in section_links
        ]

        return {
            "audio_links": audio_links,
            "duration": durations,
            "file_name": file_paths,
        }

    def _get_title(self, driver):
        title_elements = driver.find_elements(
            By.CSS_SELECTOR, self.config.title_selector
        )
        titles = [element.text for element in title_elements]

        return titles

    def _get_tags(self, driver):
        tags_container = driver.find_elements(
            By.CSS_SELECTOR, self.config.tag_selector)

        tags = [tag.text for tag in tags_container]

        return tags

    def download_tracks(self, links):
        folder = self.config.outpath

        if os.path.exists(folder) is not True:
            os.mkdir(folder)

        for link in tqdm(links):
            download_element(link, folder)


tori = FreeSoundMusicScraper()
metadata = tori.page_crawler()
print("link crawling complete")

tori.download_tracks(metadata["links"])
print("Download complete")

scraped_df = pd.DataFrame(metadata)
scraped_df.drop()

scraped_df.to_csv("metadata.csv")
