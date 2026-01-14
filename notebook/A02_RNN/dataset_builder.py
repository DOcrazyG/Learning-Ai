import torch
from config import Config
from data_processor import DataProcessor
from torch.utils.data import Dataset

config = Config()


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


if __name__ == "__main__":
    data_processor = DataProcessor(config.file_path)
    data_processor.run()
    print(data_processor.datas[2000])
    dataset = Eng2FraDataset(data_processor)
    print(dataset[2000])
