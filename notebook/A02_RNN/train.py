import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from data_preprocess import DataProcessor, Eng2FraDataset, build_dataloader
from models import Decoder, Encoder, Seq2Seq


def init_model(config, src_vocab_size, tgt_vocab_size):
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    ).to(config.device)

    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    ).to(config.device)

    model = Seq2Seq(encoder, decoder).to(config.device)

    return model


def train_one_epoch(model, dataloader, optimizer, criterion, epoch_idx, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    for batch_idx, (eng_tensors, fra_tensors) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_loss = 0

        # Get batch size
        batch_size = eng_tensors.size(0)

        # Encode the input sequence
        _, hidden = model.encoder(eng_tensors)

        # Initialize decoder input with <BOS> token (index 0)
        decoder_input = torch.tensor([[0] for _ in range(batch_size)], device=eng_tensors.device, dtype=torch.long)

        # Get target sequence (remove <BOS>)
        decoder_target = fra_tensors[:, 1:]
        seq_length = decoder_target.size(1)

        # Use teacher forcing for training
        for t in range(seq_length):
            # Forward pass through decoder
            decoder_output, hidden = model.decoder(decoder_input, hidden)

            # Calculate loss for this time step
            loss = criterion(decoder_output, decoder_target[:, t].long())
            batch_loss += loss

            # Decide whether to use teacher forcing or not
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                # Use actual target token as next input
                decoder_input = decoder_target[:, t].unsqueeze(1)
            else:
                # Use predicted token as next input (greedy decoding)
                predicted_token = torch.argmax(decoder_output, dim=-1).unsqueeze(1)
                decoder_input = predicted_token

        # Backward pass and optimization
        batch_loss = batch_loss / seq_length
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch_idx}, Batch: {batch_idx}, Loss: {batch_loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(config):
    print("Loading data...")
    data_processor = DataProcessor(config.file_path)
    data_processor.run()

    dataset = Eng2FraDataset(data_processor)
    dataloader = build_dataloader(dataset)

    print(f"Data loaded. Dataset size: {len(dataset)}")
    print(f"English vocabulary size: {data_processor.eng_vacab_length}")
    print(f"French vocabulary size: {data_processor.fra_vacab_length}")
    print("Initializing model...")
    model = init_model(config, data_processor.eng_vacab_length, data_processor.fra_vacab_length)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=2)  # ignore <PAD>

    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters())}")

    best_train_loss = float("inf")
    patience_counter = 0
    patience = 3

    print("Starting training...")
    for epoch in range(config.epochs):
        # one epoch
        train_loss = train_one_epoch(model, dataloader, optimizer, criterion, epoch)
        print(f"Epoch [{epoch + 1}/{config.epochs}], Train Loss: {train_loss:.4f}")
        # save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
            print(f"New best model saved with training loss: {train_loss:.4f}")
        else:
            patience_counter += 1

        # early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

    print("Training completed!")


if __name__ == "__main__":
    config = Config()
    train(config)
