#!/usr/bin/env python
# coding: utf-8

# # 加载模型和预定义参数

# In[1]:


# 模型部分
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
import argparse
import time
from datetime import timedelta

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--bert_path', type = str, default = 'ERNIE_pretrain')
parser.add_argument('--data_type', type = str, default = 'Origindata')
parser.add_argument('--rnn_hidden', type = int, default = 256)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--train_file', type= str, default = 'train')

opt = parser.parse_args()

class Config(object):

    """配置参数"""
    def __init__(self):
        self.class_list = [0,1,2,3]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 500                           # 若超过500iterations效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                      # 类别数
        self.num_epochs = 10                                         # epoch数
        self.batch_size = opt.batch_size                                         # mini-batch大小
        self.pad_size = 150                                      # 每句话处理成的长度(短填长切)
        self.bert_path = opt.bert_path
        self.data_type = opt.data_type
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.train_file = opt.train_file
        
        self.hidden_size = 768
        self.dropout = 0
        self.rnn_hidden = opt.rnn_hidden
        self.num_layers = 2
    
class ERNIE_RNN(nn.Module):
    def __init__(self, config):
        super(ERNIE_RNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        hidden_out, _ = self.lstm(encoder_out)
        out = self.fc_rnn(hidden_out[:, -1, :])  # 句子最后时刻的 hidden state
        return hidden_out, out


# # 读数据
PAD, CLS = '[PAD]', '[CLS]'  # padding符号

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def load_dataset(config, df, df_type, pad_size=150):
    '''
    读取数据
    df: 需要读取的数据
    df_type: "train", "dev", "test"
    pad_size: 每句话处理成的长度(短填长切)
    '''
    print("loading {}".format(df_type))
#     df = df.sample(frac=1.0, random_state=42)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    contents = []
    for index, row in df.iterrows():
        content = row["text"].strip()
        token = tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size

        if df_type == "test": 
            contents.append((token_ids, seq_len, mask))
        else:
            label = int(row["label"])
            contents.append((token_ids, seq_len, mask, label))        
    return contents


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

class TestDatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, mask)
    
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_test_iterator(dataset, config):
    iter = TestDatasetIterater(dataset, config.batch_size, config.device)
    return iter


import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
from pytorch_pretrained.optimization import BertAdam

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

# 训练model
def train(config, model, train_iter, dev_iter, learning_rate):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            hidden_outputs, outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), 'saved_dict/checkpoint_CMID.ckpt')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, dev_iter)


# 在验证集loss达到最小后500batch内没有找到更小loss时，输出验证集的预测指标和混淆矩阵
def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load('saved_dict/checkpoint_CMID.ckpt'))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, predict_all, probability_all = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

#     with open(f"CMID/{config.data_type}/{config.bert_path}_predict.txt", "w", encoding="utf-8") as f:
#         for probability in probability_all:
#             print(probability.shape)
#             probs = ''
#             for i in range(probability.shape[0]):
#                 probs += '\t' + str(probability[i])
#             f.write(probs + "\n")

# 迭代后计算并打印出当前iteration上训练集的loss/accurancy和测试集的loss/accurancy
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    probability_all = []
    
    with torch.no_grad():
        for texts, labels in data_iter:
            hidden_outputs, outputs = model(texts)
            
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
            probability = torch.softmax(outputs, dim=1)
            probability = np.round(probability.cpu().detach().numpy(), 4)
            probability_all.extend(probability)
            
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        target_names = [str(i) for i in config.class_list]
        report = metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        
        return acc, loss_total / len(data_iter), report, confusion, predict_all, probability_all
    return acc, loss_total / len(data_iter)


import numpy as np
import pandas as pd
from importlib import import_module
# def run(learning_rate, model_name):
# print("learning_rate = {}".format(learning_rate))
learning_rate = 3e-5
model_name = 'ERNIE_RNN'
config = Config()
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
df_train = pd.read_csv(f"CMID/{config.data_type}/{config.train_file}.txt", delimiter="\t", header = None)
df_train.columns = ["label", "text"]
train_data = load_dataset(config, df_train, "train", pad_size=150)

df_dev = pd.read_csv(f"CMID/Origindata/dev.txt", delimiter="\t", header = None)
df_dev.columns = ["label", "text"]
dev_data = load_dataset(config, df_dev, "dev", pad_size=150)

train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)

df_test = pd.read_csv(f"CMID/Origindata/test.txt", delimiter="\t", header = None)
df_test.columns = ["label", "text"]
test_data = load_dataset(config, df_test, "train", pad_size=150)

test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)


# train
###释放CUDA内存
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
if model_name == "ERNIE":
    model = ERNIE(config).to(config.device)
elif model_name == "ERNIE_RNN":
    model = ERNIE_RNN(config).to(config.device)
#     print(model)

time_train_start = time.time()
train(config, model, train_iter, dev_iter, learning_rate)
print('train spending: ', time.time()-time_train_start)

time_test_start = time.time()
test(config, model, test_iter)
print('test spending: ', time.time()-time_test_start)

if config.data_type == 'Origindata':
    num_aug = 1
else:
    num_aug = config.train_file[-1]


def get_embed(config, model, data_iter):
    model.eval()
    hidden_outputs_list = []

    with torch.no_grad():
        for texts, labels in data_iter:
            hidden_outputs, outputs = model(texts)
            hidden_outputs = hidden_outputs.cpu().detach().numpy()
            hidden_outputs_list.append(hidden_outputs)
    hidden_outputs_all = np.concatenate(hidden_outputs_list, axis = 0)
    return hidden_outputs_all


def save_embed(config, model, data_iter, df, file_name):
    hidden_outputs_all = get_embed(config, model, data_iter)
    df_num = len(df)
    while hidden_outputs_all.shape[0] != df_num:
        hidden_outputs_all = get_embed(config, model, data_iter)
    np.save(file_name, hidden_outputs_all)
    print("have saved bert embed to ", file_name)
    print(hidden_outputs_all.shape)


save_embed(config, model, dev_iter, df_dev, f"CMID/{config.data_type}/{config.bert_path[:-9]}_dev_{num_aug}_dim_{config.rnn_hidden}_batch_{config.batch_size}.npy")
save_embed(config, model, test_iter, df_test, f"CMID/{config.data_type}/{config.bert_path[:-9]}_test_{num_aug}_dim_{config.rnn_hidden}_batch_{config.batch_size}.npy")
if config.data_type == 'Origindata':
    save_embed(config, model, train_iter, df_train, f"CMID/{config.data_type}/{config.bert_path[:-9]}_{config.train_file}_{num_aug}_dim_{config.rnn_hidden}_batch_{config.batch_size}.npy")
else:
    save_embed(config, model, train_iter, df_train, f"CMID/{config.data_type}/{config.bert_path[:-9]}_{config.train_file}_dim_{config.rnn_hidden}_batch_{config.batch_size}.npy") 

time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)
