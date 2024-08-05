import librosa
import os
import pydub
import numpy as np


def mp3_to_wav(file: str):
    outpath = os.path.basename(file).split(".")[0]
    outpath = f"{outpath}.wav"
    sound = pydub.AudioSegment.from_mp3(file)
    sound.export(outpath)

    return outpath


# sample_url_fs = 'https://cdn.freesound.org/previews/541/541210_10912485-lq.mp3'
