import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True
        )

    def forward(self, eng_tensor):
        # eng_tensor: [batch_size, seq_len]
        # embeds: [batch_size, seq_len, embed_size]
        embeds = self.embedding(eng_tensor)
        # output: [batch_size, seq_len, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        output, hidden = self.rnn(embeds)
        return output, hidden
