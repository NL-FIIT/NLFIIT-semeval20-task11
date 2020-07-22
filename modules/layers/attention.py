import torch
from torch import nn
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SelfAttention(nn.Module):
    """
    Core of the code comes from https://github.com/cbaziotis/ntua-slp-semeval2018/blob/master/modules/nn/attention.py
    """

    def __init__(self, attention_size, dropout=.0):
        super(SelfAttention, self).__init__()

        modules = []
        modules.append(nn.Linear(attention_size, 1))
        modules.append(nn.Tanh())
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask):
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        masked_scores = scores * mask.float()
        _sums = masked_scores.sum(-1, keepdim=True)
        scores = masked_scores.div(_sums)

        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()

        return representations, scores


class SelfAttention2(nn.Module):
    def __init__(self, use_opinion, input_shape, **kwargs):
        super(SelfAttention2, self).__init__()
        self.input_dim = input_shape[0]
        self.steps = input_shape[1]
        self.use_opinion = use_opinion
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.input_dim, self.input_dim)))
        self.b = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x):
        x_tran = torch.matmul(x, self.W) + self.b
        x_transpose = x.permute((0, 2, 1))
        weights = torch.bmm(x_tran, x_transpose)
        location = np.abs(np.tile(np.array(range(self.steps)), (self.steps, 1)) - np.array(range(self.steps)).reshape(self.steps, 1))
        loc_weights = 1.0 / (location + 1e-7)
        loc_weights *= (location != 0).astype(float)
        weights *= torch.from_numpy(loc_weights).float().to(device)
        weights = torch.tanh(weights)
        weights = torch.exp(weights)
        weights = weights * (torch.eye(self.steps) == 0).float().to(device)
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-7)
        output = torch.bmm(weights, x)
        return output
