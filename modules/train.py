from tqdm import tqdm
import torch
import subprocess as sub
import os
from modules.preprocessing import task2_labels


def train(num_epochs, train_loader, valid_loader, model, optimizer, criterion, val_frequency=5):
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for samples, labels, _, lengths, _ in tqdm(train_loader):
            optimizer.zero_grad()
            output, mask = model(samples, lengths)
            loss = criterion(output.view(-1, 2), labels.view(-1))
            # loss = model.crf.compute_loss(output, labels, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {}: training loss = {}'.format(epoch, running_loss))
        if epoch % val_frequency == 0:
            predict(model, valid_loader, epoch)


def predict(model, test_loader, epoch):
    with torch.no_grad():
        model.eval()
        for samples, _, spans, lengths, file_names in test_loader:
            output, mask = model(samples, lengths)
            # torch.save(model.state_dict(), 'models/repoch{}.pt'.format(epoch))
            output = output.argmax(-1).tolist()
            # output = model.crf.forward(output, mask)
            chunks = iter(output)
            data = []
            for span, file_name in zip(spans, file_names):
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
        os.remove("predictions.txt")


def train_task2(num_epochs, train_loader, valid_loader, model, optimizer, criterion, val_frequency=5):
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for samples, labels, _, lengths, _ in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(samples, lengths)
            loss = criterion(output.view(-1, model.output_dim), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {}: training loss = {}'.format(epoch, running_loss))
        if epoch % val_frequency == 0:
            predict_task2(model, valid_loader, epoch)


def predict_task2(model, test_loader, epoch):
    with torch.no_grad():
        model.eval()
        data = []
        # torch.save(model.state_dict(), 'models/t2epoch{}.pt'.format(epoch))
        for samples, _, spans, lengths, file_names in tqdm(test_loader):
            output = model(samples, lengths)
            output = output.argmax(-1)
            predictions = output.data.tolist()
            for file_name, prediction, span in zip(file_names, predictions, spans):
                data.append("{}\t{}\t{}\t{}\n".format(file_name, task2_labels[prediction], span[0], span[1]))
        with open('prediction.txt'.format(epoch), 'w') as file:
            file.writelines(data)
        sub.call(["python3", "tools/task-TC_scorer.py", "-s", "prediction.txt",
                  "-r", "data/labels/dev-task-TC.labels", "-p",
                  "tools/data/propaganda-techniques-names-semeval2020task11.txt"])
        os.remove("prediction.txt")


def train_task2_bert(num_epochs, train_loader, valid_loader, model, optimizer, val_frequency):
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for samples, labels, _, masks, _ in tqdm(train_loader):
            optimizer.zero_grad()
            loss, output = model(samples, masks, labels)
            x = model.softmax(output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {}: training loss = {}'.format(epoch, running_loss))
        if epoch % val_frequency == 0:
            predict_task2_bert(model, valid_loader, epoch)


def predict_task2_bert(model, test_loader, epoch):
    with torch.no_grad():
        model.eval()
        data = []
        for samples, _, spans, masks, file_names in tqdm(test_loader):
            output = model(samples, masks, None)
            output = output[0].argmax(-1)
            predictions = output.data.tolist()
            for file_name, prediction, span in zip(file_names, predictions, spans):
                data.append("{}\t{}\t{}\t{}\n".format(file_name, task2_labels[prediction], span[0], span[1]))
        with open('prediction{}.txt'.format(epoch), 'w') as file:
            file.writelines(data)
