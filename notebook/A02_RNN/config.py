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
    embed_size = 64
    hidden_size = 128
    num_layers = 1

    # train
    epochs = 10
    lr = 0.001

    # predict
    max_len = 50
