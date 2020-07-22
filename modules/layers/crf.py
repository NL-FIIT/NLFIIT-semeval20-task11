from torchcrf import CRF
from torch import nn


class CRFLayer(nn.Module):
    def __init__(self, num_tags, batch_first=True):
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags
        self.crf = CRF(num_tags, batch_first=batch_first)

    def forward(self, inputs, mask):
        return self.crf.decode(inputs, mask.byte())

    def compute_loss(self, inputs, labels, mask):
        return -self.crf(inputs, labels, mask.byte(), reduction='mean')
