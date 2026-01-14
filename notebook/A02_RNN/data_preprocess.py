# download dataset from https://www.manythings.org/anki/fra-eng.zip
# unzip and put it in A02_RNN/data

import re
from typing import Dict, List

import torch
from config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

config = Config()


class DataProcessor(object):
    def __init__(self, file_path: str):
        self.file_path = file_path

        self.datas: List[List[str]] = []
        self.eng_vacab_length: int = 4
        self.eng_word2idx: Dict[str, int] = {
            "<BOS>": 0,
            "<EOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3,
        }
        self.eng_idx2word: Dict[int, str] = {}

        self.fra_vacab_length: int = 4
        self.fra_word2idx: Dict[str, int] = {
            "<BOS>": 0,
            "<EOS>": 1,
            "<PAD>": 2,
            "<UNK>": 3,
        }
        self.fra_idx2word: Dict[int, str] = {}

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        # replace non-breaking spaces with a single space
        text = text.replace("\u202f", " ").replace("\xa0", " ")
        # add spaces around punctuation
        text = re.sub(r"([,.!?])", r" \1", text)
        # clean extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _load_data(self):
        with open(self.file_path, "r", encoding="utf-8") as fr:
            lines = fr.read().strip().split("\n")
            for line in lines:
                data = line.split("\t")[:2]
                cleaned_data = [self._normalize_text(text) for text in data]
                self.datas.append(cleaned_data)

    def _build_vocab(self):
        for data in self.datas:
            for eng_word in data[0].split(" "):
                if eng_word not in self.eng_word2idx:
                    self.eng_word2idx[eng_word] = self.eng_vacab_length
                    self.eng_vacab_length += 1
            for fra_word in data[1].split(" "):
                if fra_word not in self.fra_word2idx:
                    self.fra_word2idx[fra_word] = self.fra_vacab_length
                    self.fra_vacab_length += 1
        self.eng_idx2word = {idx: word for word, idx in self.eng_word2idx.items()}
        self.fra_idx2word = {idx: word for word, idx in self.fra_word2idx.items()}

    def run(self):
        self._load_data()
        self._build_vocab()


class Eng2FraDataset(Dataset):
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor

    def __len__(self):
        return len(self.data_processor.datas)

    def __getitem__(self, idx):
        eng_sentence, fra_sentence = self.data_processor.datas[idx]

        eng_words = eng_sentence.split(" ")
        eng_words.append("<EOS>")
        fra_words = fra_sentence.split(" ")
        fra_words = ["<BOS>"] + fra_words
        fra_words.append("<EOS>")

        eng_idxs = [self.data_processor.eng_word2idx[word] for word in eng_words]
        fra_idxs = [self.data_processor.fra_word2idx[word] for word in fra_words]
        eng_tensor = torch.tensor(eng_idxs, device=config.device, dtype=torch.int32)
        fra_tensor = torch.tensor(fra_idxs, device=config.device, dtype=torch.int32)

        return eng_tensor, fra_tensor


def collate_fn(batch):
    eng_tensors = [data[0] for data in batch]
    fra_tensors = [data[1] for data in batch]
    # 2 means <PAD>
    eng_tensors = pad_sequence(eng_tensors, batch_first=True, padding_value=2)
    fra_tensors = pad_sequence(fra_tensors, batch_first=True, padding_value=2)
    return eng_tensors, fra_tensors


def build_dataloader(dataset: Eng2FraDataset):
    data_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    return data_loader
