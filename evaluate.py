import torch
from modules.preprocessing import load_task1, load_task2, task2_labels
from modules.dataloader import Loader, collate_fn, collate_fn_task2
from torch.utils.data import DataLoader
from modules.model import Classifier, Classifier2
from modules.layers.embedding import ElmoEmbedding


import subprocess as sub

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_task1():
    val_samples, val_labels, val_spans, val_file_names = load_task1('data/dev-articles', 'data/labels/dev-task-SI.labels', 'treebank')

    val_loader = Loader(val_samples, val_labels, val_spans, val_file_names)

    batch_size = 64

    valid_loader = DataLoader(val_loader, batch_size=batch_size, collate_fn=collate_fn)

    ensemble_models = ['models/epoch3.pt', 'models/epoch4.pt', 'models/epoch5.pt', 'models/epoch6.pt', 'models/epoch7.pt']

    predictions = []

    encoder_params = {'input_size': 1024, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.0, 'bidirectional': True}

    embedding = ElmoEmbedding()

    for em in ensemble_models:
        preds, val_spans, val_file_names = [], [], []
        model = Classifier(embedding, encoder_params, 2).to(device)
        model.load_state_dict(torch.load(em, map_location=device))
        with torch.no_grad():
            model.eval()
            for samples, _, spans, lengths, file_names in valid_loader:
                output, mask = model(samples, lengths)
                preds.append(output)
                val_spans.append(spans)
                val_file_names.append(file_names)
        predictions.append(preds)

    for i in range(len(predictions[0])):
        out = torch.sum(torch.stack([pred[i] for pred in predictions], 0), 0).argmax(-1).tolist()
        chunks = iter(out)
        data = []
        for span, file_name in zip(val_spans[i], val_file_names[i]):
            chunk = next(chunks)
            curr_span = ()
            flag = False
            for j, s in enumerate(span):
                if chunk[j] == 0 and flag:
                    flag = False
                    data.append([file_name, curr_span])
                    curr_span = ()
                    continue
                if chunk[j] == 1 and flag:
                    curr_span = (curr_span[0], s[1])
                    continue
                if chunk[j] == 1 and not flag:
                    flag = True
                    curr_span = (s[0], s[1])
                    continue
            if flag:
                data.append([file_name, curr_span])

        with open('predictions.txt', 'a') as f:
            for d in data:
                f.write('{}\t{}\t{}\n'.format(d[0], d[1][0], d[1][1]))
    sub.call(["python3", "tools/task-SI_scorer.py", "-s", "predictions.txt", "-r", "data/labels/reference-dev"])


def evaluate_task2():
    #val_samples, val_labels, val_spans, val_file_names = load_task2('data/test-articles', 'data/labels/test-task-TC-template.out', 'punct')
    val_samples, val_labels, val_spans, val_file_names = load_task2('data/dev-articles', 'data/labels/dev-task-TC-template.out', 'punct')

    val_loader = Loader(val_samples, val_labels, val_spans, val_file_names)

    batch_size = 64

    valid_loader = DataLoader(val_loader, batch_size=batch_size, collate_fn=collate_fn_task2)

    ensemble_models = ['models/t2epoch1.pt']

    predictions = []
    val_spans = []
    val_file_names = []

    encoder_params = {'input_size': 1024, 'hidden_size': 512, 'num_layers': 1, 'dropout': 0.3, 'bidirectional': True}

    embedding = ElmoEmbedding()

    for em in ensemble_models:
        preds, val_spans, val_file_names = [], [], []
        model = Classifier2(embedding, encoder_params, 14).to(device)
        model.load_state_dict(torch.load(em, map_location=device))
        with torch.no_grad():
            model.eval()
            for samples, _, spans, lengths, file_names in valid_loader:
                output = model(samples, lengths)
                preds.append(output)
                val_spans.append(spans)
                val_file_names.append(file_names)
        predictions.append(preds)

    data = []
    for i in range(len(predictions[0])):
        out = torch.sum(torch.stack([pred[i] for pred in predictions], 0), 0).argmax(-1).tolist()


        for pr, span, file_name in zip(out, val_spans[i], val_file_names[i]):
            data.append("{}\t{}\t{}\t{}\n".format(file_name, task2_labels[pr], span[0], span[1]))
    with open('predictions_t2.txt', 'w') as file:
        file.writelines(data)
    sub.call(["python3", "tools/task-TC_scorer.py", "-s", "predictions_t2.txt",
                  "-r", "data/labels/dev-task-TC.labels", "-p",
                  "tools/data/propaganda-techniques-names-semeval2020task11.txt"])

