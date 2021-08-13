import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from SSL_loss import SSL


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def val_set(train_loader, dev_loader, test_loader, model, epochs,lr, lr_decay_factor, lr_decay_step_size, weight_decay, save_file_name, file_root, early_stop = 30, model_name='ERNIE', aug_num=1, num_class=4):

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    best_val_loss, best_dev_acc, best_test_acc = float('inf'), 0, 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader)
        val_loss = eval_loss(model, dev_loader)
        val_acc = eval_acc(model, dev_loader)
        test_acc  = eval_acc(model, test_loader)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        duration = t_end - t_start

        if best_val_loss > val_loss:
            best_epoch = epoch
            best_dev_acc = val_acc
            best_val_loss = val_loss

        if epoch % 5 == 0:
            print('Epoch: {:03d}, Train Loss: {:.4f}, Dev Loss: {:.4f}, Dev Acc: {:3f}, Test Acc: {:.3f}, Duration: {:.3f}'
                  .format(epoch, train_loss, val_loss, val_acc, test_acc, duration))

        if epoch - best_epoch > early_stop:
            print('early stop')
            break

        if epoch == epochs:
            best_test_acc = test_acc
            best_dev_acc = val_acc
            best_val_loss = val_loss

    if torch.cuda.is_available():
        torch.cuda.synchronize()

#     print("###### dev #######")
#     predict(model, dev_loader, f"{file_root}/{model_name}_aug_{aug_num}_dev_{save_file_name}.txt")
    print("###### test #######")
    predict(model, test_loader, f"{file_root}/{model_name}_aug_{aug_num}_test_{save_file_name}.txt", num_class)

    return best_val_loss, best_dev_acc, best_test_acc


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def predict(model, loader, file_name, num_class):
    model.eval()
    labels_all, predict_all = [], []
    f = open(file_name, "w")
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            pred = output.max(dim=1)[1]
            probability = torch.softmax(output, dim=1)
            probability = np.round(probability.cpu().detach().numpy(), 4)
            for i in range(len(data.y)):
            #     f.write(str(int(data.y[i]))+"\t"+str(int(pred[i]))+"\n")
                probs = ''
                for j in range(probability.shape[1]):
                    probs += '\t' + str(probability[i][j])
                f.write(str(int(data.y[i])) + probs + "\n")
                labels_all.append(int(data.y[i]))
                predict_all.append(int(pred[i]))

    # acc = metrics.accuracy_score(labels_all, predict_all)
    target_names = [str(i) for i in range(num_class)]
    report = metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    # print(acc)
    print(report)
#     print(confusion)




def val_set_SSL(train_loader1, dev_loader1, test_loader1, train_loader2, dev_loader2, test_loader2,
                   model, epochs, lr, lr_decay_factor, lr_decay_step_size, weight_decay, save_file_name, beta, 
                file_root, early_stop = 50,model_name='ERNIE', aug_num=1, num_class=4):

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    best_val_loss, best_dev_acc, best_test_acc = float('inf'), 0, 0
    best_epoch = 0
    
    print('start training...')
    for epoch in range(1, epochs + 1):
        train_loss = train_SSL(model, optimizer, train_loader1, train_loader2, beta)
        val_loss = eval_loss_SSL(model, dev_loader1, dev_loader2, beta)
        val_acc = eval_acc_SSL(model, dev_loader1, dev_loader2)
        test_acc = eval_acc_SSL(model, test_loader1, test_loader2)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        t_end = time.perf_counter()
        duration = t_end - t_start

        if best_val_loss > val_loss:
            best_epoch = epoch
            best_dev_acc = val_acc
            best_val_loss = val_loss

        if epoch % 5 == 0:
            print(
                'Epoch: {:03d}, Train Loss: {:.4f}, Dev Loss: {:.4f}, Dev Acc: {:3f}, Test Acc: {:.3f}, Duration: {:.3f}'
                .format(epoch, train_loss, val_loss, val_acc, test_acc, duration))

        if epoch - best_epoch > early_stop:
            print('early stop')
            break

        if epoch == epochs:
            best_test_acc = test_acc
            best_dev_acc = val_acc
            best_val_loss = val_loss
    
    print('training spending: ', time.perf_counter()-t_start)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

#     print("###### dev #######")
#     predict_concat(model, dev_loader1, dev_loader2, f"{file_root}/{model_name}_aug_{aug_num}_dev_{save_file_name}.txt")
    print("###### test #######")
    t_predict = time.perf_counter()
    predict_SSL(model, test_loader1, test_loader2, f"{file_root}/{model_name}_aug_{aug_num}_test_{save_file_name}.txt", num_class)
    print('test spending: ', time.perf_counter()-t_predict)
    
    return best_val_loss, best_dev_acc, best_test_acc


def train_SSL(model, optimizer, loader1, loader2, beta):
    model.train()

    total_loss = 0
    for data in zip(loader1, loader2):
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        optimizer.zero_grad()
        embed1, embed2, out = model(data1, data2)
        loss1 = F.nll_loss(out, data1.y.view(-1))* num_graphs(data1)
        loss2 = SSL(embed1, embed2)
        loss = loss1 + beta * loss2
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader1.dataset)


def eval_acc_SSL(model, loader1, loader2):
    model.eval()

    correct = 0
    for i, data1 in enumerate(loader1):
        for j, data2 in enumerate(loader2):
            if i == j:
                data1 = data1.to(device)
                data2 = data2.to(device)
                with torch.no_grad():
                    embed1, embed2, out = model(data1, data2)
                    pred = out.max(1)[1]
                correct += pred.eq(data1.y.view(-1)).sum().item()
    return correct / len(loader1.dataset)


def eval_loss_SSL(model, loader1, loader2, beta):
    model.eval()

    loss1 = 0
    loss2 = 0
    for data in zip(loader1, loader2):
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        with torch.no_grad():
            embed1, embed2, out = model(data1, data2)
        
        loss1 += F.nll_loss(out, data1.y.view(-1), reduction='sum').item() # * num_graphs(data1)
        loss2 += SSL(embed1, embed2).item()
    loss = loss1 + beta * loss2
        
    return loss / len(loader1.dataset)


def predict_SSL(model, loader1, loader2, file_name, num_class):
    model.eval()
    labels_all, predict_all = [], []
    f = open(file_name, "w")
    for data in zip(loader1, loader2):
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        with torch.no_grad():
            embed1, embed2, output = model(data1, data2)
            pred = output.max(dim=1)[1]
            probability = torch.softmax(output, dim=1)
            probability = np.round(probability.cpu().detach().numpy(), 4)
            for i in range(len(data1.y)):
                probs = ''
                for j in range(probability.shape[1]):
                    probs += '\t' + str(probability[i][j])
                f.write(str(int(data1.y[i])) + probs + "\n")
                labels_all.append(int(data1.y[i]))
                predict_all.append(int(pred[i]))

    # acc = metrics.accuracy_score(labels_all, predict_all)
    target_names = [str(i) for i in range(num_class)]
    report = metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    print(report)