from modules.preprocessing import load_task1
from modules.dataloader import Loader, collate_fn
from torch.utils.data import DataLoader
from modules.layers.embedding import GloveEmbedding, ElmoEmbedding
from modules.model import Classifier
from torch.optim import Adam
from modules.train import train
import torch.nn as nn
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

samples, labels, spans, file_names = load_task1('data/train-articles', 'data/labels/train-task1-SI.labels', 'treebank')
val_samples, val_labels, val_spans, val_file_names = load_task1('data/dev-articles', 'data/labels/dev-task-SI.labels', 'treebank')

tr_loader = Loader(samples, labels, spans, file_names)
val_loader = Loader(val_samples, val_labels, val_spans, val_file_names)

batch_size = 64

train_loader = DataLoader(tr_loader, batch_size=batch_size, collate_fn=collate_fn)
valid_loader = DataLoader(val_loader, batch_size=batch_size, collate_fn=collate_fn)

encoder_params = {'input_size': 1024, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.1, 'bidirectional': True}

embedding = ElmoEmbedding()
model = Classifier(embedding, encoder_params, 2).to(device)
optimizer = Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

train(20, train_loader, valid_loader, model, optimizer, criterion, val_frequency=1)

