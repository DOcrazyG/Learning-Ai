from config import Config
from data_processor import DataProcessor
from dataset_builder import Eng2FraDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

config = Config()


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


if __name__ == "__main__":
    data_processor = DataProcessor(config.file_path)
    data_processor.run()
    dataset = Eng2FraDataset(data_processor)
    data_loader = build_dataloader(dataset)
    for batch in data_loader:
        print(batch)
        break
