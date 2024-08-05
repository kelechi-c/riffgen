class scraper_config:
    sdd_dataset_url = "https://zenodo.org/records/10072001/files/audio.zip?download=1"
    num_files_target = 1000
    outpath = "music"
    num_pages = 10

    # freesound scraping configs
    freesound_url = "https://freesound.org/"
    freesound_url_for_music = (
        "https://freesound.org/browse/tags/?f=tag%3A%22music%22&page=1#sound"
    )
    player_id = "bw-player"
    mp3_attr = "data-mp3"
    duration_attr = "data-duration"
    container_class = "bw-search__result"
    mp3link_selector = "audio > source"
    title_selector = "h5.v-spacing-1 > a.bw-link--black"
    tag_selector = "a.tag-container"
    pagination_links = "bw-pagination_container"
