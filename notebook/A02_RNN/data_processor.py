# download dataset from https://www.manythings.org/anki/fra-eng.zip
# unzip and put it in A02_RNN/data

import re
from typing import Dict, List


class DataProcessor(object):
    datas: List[List[str]] = []

    eng_vacab_length: int = 4
    eng_word2idx: Dict[str, int] = {"<BOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
    eng_idx2word: Dict[int, str] = {}

    fra_vacab_length: int = 4
    fra_word2idx: Dict[str, int] = {"<BOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
    fra_idx2word: Dict[int, str] = {}

    def __init__(self, file_path: str):
        self.file_path = file_path

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


if __name__ == "__main__":
    from config import Config

    config = Config()

    data_processor = DataProcessor(config.file_path)
    data_processor.run()
    print(f"datas_length:{len(data_processor.datas)}")
    print(f"eng_vacab_length:{data_processor.eng_vacab_length}")
    print(f"eng_word2idx:{data_processor.eng_word2idx}")
    print(f"fra_vacab_length:{data_processor.fra_vacab_length}")
    print(f"fra_word2idx:{data_processor.fra_word2idx}")
