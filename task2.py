from modules.preprocessing import load_task2, load_task2_bert
from modules.dataloader import Loader, collate_fn_task2, collate_fn_bert
from torch.utils.data import DataLoader
from modules.layers.embedding import ElmoEmbedding, GloveEmbedding
from modules.model import Classifier2, BertClassifier
from torch.optim import Adam
from modules.train import train_task2, train_task2_bert
import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

samples, labels, spans, file_names = load_task2('data/train-articles', 'data/labels/train-task2-TC.labels', 'punct')
val_samples, val_labels, val_spans, val_file_names = load_task2('data/dev-articles', 'data/labels/dev-task-TC-template.out', 'punct')

tr_loader = Loader(samples, labels, spans, file_names)
val_loader = Loader(val_samples, val_labels, val_spans, val_file_names)

batch_size = 32

train_loader = DataLoader(tr_loader, batch_size=batch_size, collate_fn=collate_fn_task2)
valid_loader = DataLoader(val_loader, batch_size=batch_size, collate_fn=collate_fn_task2)

encoder_params = {'input_size': 1024, 'hidden_size': 512, 'num_layers': 1, 'dropout': 0.2, 'bidirectional': True}

embedding = ElmoEmbedding()
model = Classifier2(embedding, encoder_params, 14).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

train_task2(20, train_loader, valid_loader, model, optimizer, criterion, val_frequency=1)
