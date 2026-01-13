from typing import List

from torch.utils.data import Dataset


class DataLoaderBuilder(Dataset):
    def __init__(self, datas: List[List[str]]):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]
