import torch.nn as nn
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import BertModel, BertConfig
import codecs

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ElmoEmbedding(nn.Module):

    def __init__(self, dropout=0.):
        super(ElmoEmbedding, self).__init__()
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1).to(device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sentences):
        # sentences = list(map(lambda sentence: [token.lower() for token in sentence], sentences))
        # Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters (len(batch), max sentence length, max word length).
        character_ids = batch_to_ids(sentences).to(device)

        embedded = self.elmo(character_ids)
        # A num_output_representations list of ELMo representations for the input sequence. Each representation is shape (batch_size, timesteps, embedding_dim)
        embeddings = self.dropout(embedded['elmo_representations'][0])
        # Shape (batch_size, timesteps) long tensor with sequence mask.
        mask = embedded['mask']
        return embeddings, mask


def load_glove_file(path):
    glove_dict = {}
    with codecs.open(path, 'r') as file:
        for line in file:
            split_line = line.split(' ')
            token = split_line[0]
            embedding = torch.FloatTensor([float(value) for value in split_line[1:]])
            glove_dict[token] = embedding
    return glove_dict


def create_mask(batch_size, max_length, lengths):
    mask = torch.zeros(batch_size, max_length)
    for index, length in enumerate(lengths):
        mask[index, :length] = 1
    return mask


class GloveEmbedding(nn.Module):
    def __init__(self, glove_file_path='glove.840B.300d.txt',  dropout=.2):
        super(GloveEmbedding, self).__init__()
        self.glove_dict = load_glove_file(glove_file_path)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences):
        max_length = max(map(len, sentences))
        batch_size = len(sentences)
        lengths = list(map(len, sentences))
        batch_tensors = []
        for sentence in sentences:
            embed_tensors = []
            for token in sentence:
                vec = self.glove_dict.get(token, torch.FloatTensor(300).uniform_(-0.05, 0.05))
                embed_tensors.append(vec)
            pad_length = max_length - len(sentence)
            embedding = torch.stack(embed_tensors)
            if pad_length:
                padding = torch.zeros(pad_length, 300)
                embedding = torch.cat([embedding, padding], dim=0)
            batch_tensors.append(embedding)

        embeddings = torch.stack(batch_tensors, dim=0).to(device)
        mask = create_mask(batch_size, max_length, lengths).to(device)
        return self.dropout(embeddings), mask

