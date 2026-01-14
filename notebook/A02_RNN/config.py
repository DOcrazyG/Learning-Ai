import torch


class Config:
    file_path = "./data/fra.txt"
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    max_len = 40
    batch_size = 64
