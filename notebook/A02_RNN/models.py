import torch
import torch.nn as nn
from config import Config

config = Config()


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True
        )

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
            embed_size + hidden_size,  # teach forcing
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, fra_tensor, state):
        # embeds: [batch_size, dc_seq_len, dc_embed_size]
        embeds = self.embedding(fra_tensor)
        # encoder_output: [batch_size, ec_seq_len, ec_hidden_size]
        # hn: [ec_num_layers, batch_size, ec_hidden_size]
        encoder_output, hn = state
        # context: [batch_size, 1, ec_hidden_size]
        context = encoder_output[:, -1:, :]
        # broadcast context to [batch_size, ec_seq_len, hidden_size]
        context = context.expand(-1, embeds.size(1), -1)
        embeds_and_context = torch.cat((embeds, context), dim=-1)

        # output: [batch_size, seq_len, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        output, hidden = self.rnn(embeds_and_context, hn)
        # output: [batch_size, seq_len, vocab_size]
        output = self.fc(output)
        return output, (encoder_output, hidden)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, eng_tensor, fra_tensor):
        encoder_output, hidden = self.encoder(eng_tensor)
        output, _ = self.decoder(fra_tensor, (encoder_output, hidden))
        # output: [batch_size, seq_len, vocab_size]
        return output
