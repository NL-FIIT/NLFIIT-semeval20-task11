from torch.utils.data.dataset import Dataset
import torch
from transformers import BertTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Loader(Dataset):

    def __init__(self, samples, labels, spans, file_names):
        self.samples = samples
        self.labels = labels
        self.spans = spans
        self.file_names = file_names

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.spans[idx], self.file_names[idx]

    def __len__(self):
        return len(self.samples)


def collate_fn(data):
    samples, labels, spans, file_names = zip(*data)
    lengths = list(map(lambda s: len(s), samples))
    labels = list(map(lambda l: torch.LongTensor(l).to(device), labels))
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    return samples, labels, spans, lengths, file_names


def collate_fn_task2(data):
    samples, labels, spans, file_names = zip(*data)
    lengths = list(map(len, samples))
    labels = torch.LongTensor(labels).to(device)
    return samples, labels, spans, lengths, file_names


def collate_fn_bert(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_sents, labels, spans, file_names = zip(*data)
    lengths = list(map(len, tokenized_sents))
    labels = torch.tensor(labels).to(device)
    max_length = max(lengths) + 1
    encoded = list(map(lambda sent: tokenizer.encode_plus(sent, max_length=max_length, pad_to_max_length=True), tokenized_sents))
    samples = torch.LongTensor(list(item['input_ids'] for item in encoded)).to(device)
    masks = torch.FloatTensor(list(item['attention_mask'] for item in encoded)).to(device)
    return samples, labels, spans, masks, file_names
