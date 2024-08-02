class config:
    sdd_dataset_url = "https://zenodo.org/records/10072001/files/audio.zip?download=1"
    num_files_target = 1000

    # freesound scraping configs
    freesound_url = "https://freesound.org/"
    player_id = "bw-player"
    mp3_attr = "data-mp3"
    container_class = "bw-search__result"
    mp3link_selector = "audio > source"
    title_selector = "div.between > div > h5 > a"
