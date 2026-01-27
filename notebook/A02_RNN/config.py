import torch


class Config:
    file_path = "./data/fra.txt"
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    batch_size = 32

    # rnn hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 2

    # train
    epochs = 100
    lr = 1e-3

    # predict
    max_len = 50
