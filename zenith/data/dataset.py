import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Loads a .npy array of shape (n_sequences, seq_len) of token ids via memmap."""

    def __init__(self, npy_path: str):
        self.data = np.load(npy_path, mmap_mode="r")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx].astype(np.int64))
        x = seq[:-1]
        y = seq[1:]
        return x, y
