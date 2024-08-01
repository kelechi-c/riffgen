from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import librosa

hfdata = load_dataset()


class MusicData(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
