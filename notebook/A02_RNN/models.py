import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

config = Config()


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, eng_tensor, hidden=None):
        # eng_tensor: [batch_size, seq_len]
        # embeds: [batch_size, seq_len, embed_size]
        embeds = self.embedding(eng_tensor)

        # Dynamically determine batch size from input
        batch_size = embeds.size(0)
        if hidden is None:
            hidden = torch.zeros(
                self.rnn.num_layers,
                batch_size,
                self.rnn.hidden_size,
                device=eng_tensor.device,
            )

        # output: [batch_size, seq_len, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        output, hidden = self.rnn(embeds, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, fra_tensor, hidden):
        # embeds: [batch_size, dc_seq_len, dc_embed_size]
        embeds = self.embedding(fra_tensor)
        embeds = F.relu(embeds)
        # output: [batch_size, seq_len, vocab_size]
        # hidden: [num_layers, batch_size, hidden_size]
        output, hidden = self.rnn(embeds, hidden)
        output = self.fc(output[0])
        output = self.softmax(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, eng_tensor, fra_tensor):
        _, hidden = self.encoder(eng_tensor)
        output, hidden = self.decoder(fra_tensor, hidden)
        return output, hidden
