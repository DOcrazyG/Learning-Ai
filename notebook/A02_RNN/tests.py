from config import Config
from data_preprocess import (
    DataProcessor,
    Eng2FraDataset,
    build_dataloader,
)
from models import Encoder

config = Config()


def test_dataprocessor():
    data_processor = DataProcessor(config.file_path)
    data_processor.run()
    print(f"datas_length:{len(data_processor.datas)}")
    print(f"eng_vacab_length:{data_processor.eng_vacab_length}")
    # print(f"eng_word2idx:{data_processor.eng_word2idx}")
    print(f"fra_vacab_length:{data_processor.fra_vacab_length}")
    # print(f"fra_word2idx:{data_processor.fra_word2idx}")


def test_dataset():
    data_processor = DataProcessor(config.file_path)
    data_processor.run()
    dataset = Eng2FraDataset(data_processor)
    print(f"dataset_length:{len(dataset)}")
    print(f"dataset[0]:{dataset[0]}")


def test_dataloader():
    data_processor = DataProcessor(config.file_path)
    data_processor.run()
    dataset = Eng2FraDataset(data_processor)
    dataloader = build_dataloader(dataset)
    print(f"dataloader_length:{len(dataloader)}")
    for eng_tensors, fra_tensors in dataloader:
        print(f"eng_tensors.shape:{eng_tensors.shape}")
        print(f"fra_tensors.shape:{fra_tensors.shape}")
        break


def test_encoder():
    data_processor = DataProcessor(config.file_path)
    data_processor.run()
    dataset = Eng2FraDataset(data_processor)
    dataloader = build_dataloader(dataset)
    encoder = Encoder(
        vocab_size=data_processor.eng_vacab_length,
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    ).to(config.device)
    for eng_tensors, _ in dataloader:
        output, hidden = encoder(eng_tensors)
        print(f"output.shape:{output.shape}")
        print(f"hidden.shape:{hidden.shape}")
        break


if __name__ == "__main__":
    test_dataprocessor()
    print("-" * 60)
    test_dataset()
    print("-" * 60)
    test_dataloader()
    print("-" * 60)
    test_encoder()
