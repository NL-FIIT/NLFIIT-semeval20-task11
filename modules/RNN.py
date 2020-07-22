import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNN(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=2, dropout=0.15, bidirectional=False):
        super(RNN, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.feature_size = hidden_size * self.num_directions
        self.model = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True,
                             bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, lengths, hidden=None):
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return self.dropout(output), hidden
