class scraper_config:
    sdd_dataset_url = "https://zenodo.org/records/10072001/files/audio.zip?download=1"
    outpath = "data"
    num_pages = 500

    # freesound scraping configs
    player_id = "div.bw-player"
    mp3_attr = "data-mp3"
    duration_attr = "data-duration"
    container_class = "bw-search__result"
    mp3link_selector = "audio > source"
    title_selector = "h5.v-spacing-1 > a.bw-link--black"
    tag_selector = "a.tag-container"
    tag_container = "div.bw-player__tags-list-height"
