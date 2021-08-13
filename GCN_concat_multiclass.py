#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
from utils_multiclass import *

start=time.time()

layers = [1]
hiddens = [128]

import argparse
parser = argparse.ArgumentParser(description='GCN_concat_multiclass')
parser.add_argument('--model', type = str, default = 'ERNIE')
parser.add_argument('--dataset', type = str, default = 'CMID')
parser.add_argument('--data_type', type = str, default = 'Origindata')
parser.add_argument('--aug_num', type = int, default = 1)
parser.add_argument('--beta', type = float, default = 0.01)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--times', type = int, default = 5)

opt = parser.parse_args()
file_root = f'{opt.dataset}/{opt.data_type}/'
print(file_root)

if opt.dataset == 'CMID':
    pad_size = 150
    num_class = 4 
    weight_decay = 0.0001   #L2
elif opt.dataset == 'KUAKE-QIC':
    pad_size = 100
    num_class = 11
    weight_decay = 0.001 # L2
else:
    print('dataset does not exist!')
     

if opt.data_type == 'Origindata':
    seg_file = file_root + 'seg.json'
    pos_file = file_root + 'pos.json'
    dep_file = file_root + 'dep.json'
    label_file = file_root + "label.json"
    embed_key = f'{opt.model}_dim_256_batch_32_'
    for file in os.listdir(file_root):
        if embed_key in file:
            embed_file = file_root + file
else:
    seg_file = file_root + f'seg_{opt.aug_num}.json'
    pos_file = file_root + f'pos_{opt.aug_num}.json'
    dep_file = file_root + f'dep_{opt.aug_num}.json'
    label_file = file_root + f"label_{opt.aug_num}.json"
    embed_key = f'{opt.model}_aug_{opt.aug_num}_dim_256_batch_32_'
    for file in os.listdir(file_root):
        if embed_key in file:
            embed_file = file_root + file
            
splits = embed_file.replace('.npy', '').split('_')
test_num = int(splits[-1])
dev_num = int(splits[-2])
train_num = int(splits[-3])


# 读取数据
def get_edge_index(seg):
    node1_list, node2_list = [], []
    sentence = ''.join(seg)
    if len(sentence) > pad_size:
        sentence = sentence[0:pad_size]

    # 连续的两个字连一条边
    for i, char in enumerate(sentence):
        node1 = i
        node2 = i + 1
        if node2 < len(sentence):
            node1_list.append(node1)
            node1_list.append(node2)
            node2_list.append(node2)
            node2_list.append(node1)

    idx = 0
    for i, word in enumerate(seg):
        if len(word) > 2:
            node1 = idx
            node2 = node1 + len(word) -1
            if node2 < pad_size:
                node1_list.append(node1)
                node1_list.append(node2)
                node2_list.append(node2)
                node2_list.append(node1)
        idx += len(word)
        if idx >= pad_size: break
    edge_index = np.array([node1_list, node2_list], dtype=np.int64)
    return edge_index


def get_embed_vector(embed, seg):
    sentence = ''.join(seg)
    if len(sentence) > pad_size:
        sentence = sentence[0:pad_size]
    arr = embed[0:len(sentence)]
    return np.array(arr)


def lexicon_dataset():
    seg = json.load(open(seg_file, encoding='utf-8'))
    embed = np.load(embed_file)
    labels = json.load(open(label_file, encoding='utf-8'))

    data_list = []
    if len(seg) == len(embed) == len(labels):
        # seg: 分词, pos: 词性, dep: 句法依存

        print("#samples is ", len(seg))
        for i in range(len(seg)):
            sentence = seg[i]
            embed_vector = get_embed_vector(embed[i], sentence)
            x = torch.tensor(embed_vector, dtype=torch.float)
            y = torch.tensor([labels[i]], dtype=torch.int64)
            edge_index = get_edge_index(sentence)
            edge_index = torch.tensor(edge_index)
            data = Data(x = x, edge_index = edge_index, y = y)
            data_list.append(data)
    else:
        print('something wrong while reading data')
    return data_list


# In[5]:


class PyGLexiconDataset(InMemoryDataset):
    def __init__(self, save_root, transform=None, pre_transform=None):
        """
        :param save_root:保存数据的目录
        :param pre_transform:在读取数据之前做一个数据预处理的操作
        :param transform:在访问之前动态转换数据对象(因此最好用于数据扩充)
        """
        super(PyGLexiconDataset, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):  # 原始数据文件夹存放位置,这个例子中是随机出创建的，所以这个文件夹为空
        return ['origin_dataset']

    @property
    def processed_file_names(self):
        return ['lexicon_dataset.pt']

    def download(self):  # 这个例子中不是从网上下载的，所以这个函数pass掉
        pass

    def process(self):   # 处理数据的函数,最关键（怎么创建，怎么保存）
        data_list = lexicon_dataset()
        data_save, data_slices = self.collate(data_list) # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
        torch.save((data_save, data_slices), self.processed_file_names[0])


dataset1 = PyGLexiconDataset(save_root = file_root + f"lexicon_aug_{opt.aug_num}")
train_dataset1 = dataset1[0: train_num]
dev_dataset1 = dataset1[train_num: (train_num + dev_num)]
test_dataset1 = dataset1[(train_num + dev_num):]

train_loader1 = DataLoader(train_dataset1, batch_size=opt.batch_size, shuffle=True)
dev_loader1 = DataLoader(dev_dataset1, batch_size=opt.batch_size, shuffle=False)
test_loader1 = DataLoader(test_dataset1, batch_size=opt.batch_size, shuffle=False)



PART_OF_SPEECH = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'z', 'wp', 'ws', 'x'] #29个词性
SYNTAX = ['SBV', 'VOB', 'IOB', 'FOB', 'DBL', 'ATT', 'ADV', 'CMP', 'COO', 'POB', 'LAD', 'RAD', 'IS'] # 13个依存句法关系
SYNTAX = [i.lower() for i in SYNTAX]



def get_one_hot_vector(label_list, values):
    # 生成one-hot编码
    num_classes = len(label_list) # 设置类别的数量
    arr = [] # 需要转换的整数
    for v in values:
        arr.append(label_list.index(v.lower()))
    return np.eye(num_classes)[arr] # 将整数转为一个10位的one hot编码

def get_edge_info(dep, sign):
    node1_list, node2_list, relations = [], [], []
    for triple in dep:
        node1 = triple[0] - 1
        node2 = triple[1] - 1
        relation = triple[2]
        if relation == 'HED' or relation.lower() not in SYNTAX:
            continue
        elif node1 == -1 or node2 == -1:
            continue
        elif sign > 0 and (node1 > sign or node2 > sign):
            continue
        else:
            node1_list.append(node1)
            node1_list.append(node2)
            node2_list.append(node2)
            node2_list.append(node1)
            relations.append(relation)
    edge_attr = get_one_hot_vector(SYNTAX, relations)
    edge_index = np.array([node1_list, node2_list], dtype=np.int64)

    return edge_attr, edge_index


def get_embed_vector(embed, sentence):
    idx = 0
    arr = []
    sign = 0
    for i, word in enumerate(sentence):
        len_word = len(word)
        if idx > pad_size:
            return np.array(arr), sign
        elif idx+len_word >= pad_size: #句子的最大长度
            word_embed = embed[idx:pad_size].mean(axis=0)
            sign = i
            arr.append(word_embed)
            return np.array(arr), sign
        else:
            word_embed = embed[idx:(idx+len_word)].mean(axis=0)
            idx = idx + len_word
            arr.append(word_embed)
    return np.array(arr), sign


def syntax_dataset():
    seg = json.load(open(seg_file, encoding='utf-8'))
    pos = json.load(open(pos_file, encoding='utf-8'))
    dep = json.load(open(dep_file, encoding='utf-8'))
    embed = np.load(embed_file)
    labels = json.load(open(label_file, encoding='utf-8'))

    data_list = []
    if len(seg) == len(pos) == len(embed) == len(dep) == len(labels):
        # seg: 分词, pos: 词性, dep: 句法依存
        print("#samples is ", len(seg))
        for i in range(len(seg)):
            sentence = seg[i]
            embed_vector, sign = get_embed_vector(embed[i], sentence)
            x = torch.tensor(embed_vector, dtype=torch.float)
            y = torch.tensor([labels[i]], dtype=torch.int64)
            edge_attr, edge_index = get_edge_info(dep[i], sign)
            edge_index = torch.tensor(edge_index)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)
            data_list.append(data)
    else:
        print('something wrong while reading data')
    return data_list


class PyGSyntaxDataset(InMemoryDataset):
    def __init__(self, save_root, transform=None, pre_transform=None):
        """
        :param save_root:保存数据的目录
        :param pre_transform:在读取数据之前做一个数据预处理的操作
        :param transform:在访问之前动态转换数据对象(因此最好用于数据扩充)
        """
        super(PyGSyntaxDataset, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):  # 原始数据文件夹存放位置,这个例子中是随机出创建的，所以这个文件夹为空
        return ['origin_dataset']

    @property
    def processed_file_names(self):
        return ['syntax_dataset.pt']

    def download(self):  # 这个例子中不是从网上下载的，所以这个函数pass掉
        pass

    def process(self):   # 处理数据的函数,最关键（怎么创建，怎么保存）
        data_list = syntax_dataset()
        data_save, data_slices = self.collate(data_list) # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
        torch.save((data_save, data_slices), self.processed_file_names[0])


dataset2 = PyGSyntaxDataset(save_root = file_root + f"syntax_aug_{opt.aug_num}")
train_dataset2 = dataset2[0: train_num]
dev_dataset2 = dataset2[train_num: (train_num + dev_num)]
test_dataset2 = dataset2[(train_num + dev_num):]

train_loader2 = DataLoader(train_dataset2, batch_size=opt.batch_size, shuffle=True)
dev_loader2 = DataLoader(dev_dataset2, batch_size=opt.batch_size, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=opt.batch_size, shuffle=False)

print('Prepare data spending: ', time.time()-start)

from gcn import GCN
from gcn import GCN_concat, GCN_add
from itertools import product
times = opt.times


# # GCN_add_SSL
start = time.time()
model_name = GCN_add.__name__
for num_layers, hidden in product(layers, hiddens):
    for t in range(times):
        model = GCN_add(dataset1, dataset2, num_layers, hidden)
        print('------ {} - SSL - {} - {} - {} '.format(model_name, num_layers, hidden, t))
        save_file_name = f'{model_name}_SSL_layer_{num_layers}_hidden_{hidden}_time_{t}'

        dev_loss, dev_acc, test_acc = val_set_SSL(
            train_loader1,
            dev_loader1,
            test_loader1,
            train_loader2,
            dev_loader2,
            test_loader2,
            model,
            epochs=200,
            lr=0.005,
            lr_decay_factor=0.5,
            lr_decay_step_size=50,
            weight_decay=weight_decay,
            save_file_name= save_file_name,
            beta = opt.beta,
            file_root = file_root,
            model_name = opt.model,
            aug_num = opt.aug_num,
            num_class = num_class,
            )
    
print('Train & test model spend: ', time.time()-start)
        


# # lexicon result
# model_name = GCN.__name__
# for num_layers, hidden in product(layers, hiddens):
#     for t in range(times):
#         model = GCN(dataset1, num_layers, hidden)
#         print('------ {} - {} - {} - {}'.format(model_name, num_layers, hidden, t))
#         save_file_name = f'{model_name}_lexicon_layer_{num_layers}_hidden_{hidden}_time_{t}'
#         mmm = f"{opt.model}_aug_{opt.aug_num}_test_{save_file_name}.txt"
#         files = os.listdir(file_root)
#         if mmm in files:
#             print(f"{mmm} has existed, continue ...")
#             continue
#         dev_loss, dev_acc, test_acc = val_set(
#             train_loader1,
#             dev_loader1,
#             test_loader1,
#             model,
#             epochs=200,
#             lr=0.01,
#             lr_decay_factor=0.5,
#             lr_decay_step_size=50,
#             weight_decay=0,
#             save_file_name= save_file_name,
#             model_name = opt.model,
#             aug_num = opt.aug_num
#             )



# #syntax result
# model_name = GCN.__name__
# for num_layers, hidden in product(layers, hiddens):
#     for t in range(times):
#         model = GCN(dataset2, num_layers, hidden)
#         print('------ {} - {} - {} - {}'.format(model_name, num_layers, hidden, t))
#         save_file_name = f'{model_name}_syntax_layer_{num_layers}_hidden_{hidden}_time_{t}'
#         mmm = f"{opt.model}_aug_{opt.aug_num}_test_{save_file_name}.txt"
#         files = os.listdir(file_root)
#         if mmm in files:
#             print(f"{mmm} has existed, continue ...")
#             continue
                
#         dev_loss, dev_acc, test_acc = val_set(
#             train_loader2,
#             dev_loader2,
#             test_loader2,
#             model,
#             epochs=200,
#             lr=0.01,
#             lr_decay_factor=0.5,
#             lr_decay_step_size=50,
#             weight_decay=0,
#             save_file_name= save_file_name,
#             model_name = opt.model,
#             aug_num = opt.aug_num
#             )
        