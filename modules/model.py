from modules.RNN import RNN
from modules.layers.attention import SelfAttention, SelfAttention2
from modules.layers.crf import CRFLayer
from transformers import BertForSequenceClassification
from torch import nn


class Classifier(nn.Module):
    def __init__(self, embeddings, encoder_params, output_dim):
        super(Classifier, self).__init__()
        self.output_dim = output_dim
        self.embeddings = embeddings
        self.encoder = RNN(**encoder_params)
        self.linear = nn.Linear(self.encoder.feature_size, output_dim)
        # self.crf = CRFLayer(num_tags=2, batch_first=True)
        # self.attention = SelfAttention(self.encoder.feature_size)

    def forward(self, inputs, lengths):
        embeds, mask = self.embeddings(inputs)
        rnn_out, hidden = self.encoder(embeds, lengths)
        # rnn_out, _ = self.attention(rnn_out, mask)
        # attention = SelfAttention2(0, [rnn_out.size(-1), rnn_out.size(1)]).to('cuda')
        # rnn_out = attention(rnn_out)
        output = self.linear(rnn_out)
        return output, mask


class Classifier2(nn.Module):
    def __init__(self, embeddings, encoder_params, output_dim):
        super(Classifier2, self).__init__()
        self.output_dim = output_dim
        self.embeddings = embeddings
        self.encoder = RNN(**encoder_params)
        self.attention = SelfAttention(self.encoder.feature_size)
        self.linear = nn.Linear(self.encoder.feature_size, output_dim)

    def forward(self, inputs, lengths):
        embeds, mask = self.embeddings(inputs)
        rnn_out, hidden = self.encoder(embeds, lengths)
        representations, scores = self.attention(rnn_out, mask)
        output = self.linear(representations)
        return output

class BertClassifier(nn.Module):
    def __init__(self, output_dim):
        super(BertClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=output_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, input_ids, masks, labels):
        outputs = self.model(input_ids, labels=labels, attention_mask=masks)
        return outputs

