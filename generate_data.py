import json
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from ltp import LTP
ltp_exm = LTP()
# user_dict.txt 是词典文件， max_window是最大前向分词窗口
ltp_exm.init_dict(path="jiebaDict.txt", max_window=20)

import argparse


# root_file = '/home/admin-pku/ruiqing/bert_embed/'
root_file = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='parser sentence')
parser.add_argument('--model', type = str, default = 'ERNIE')
parser.add_argument('--dataset', type = str, default = 'CMID')
parser.add_argument('--folder', type = str, default = 'Origindata/')
parser.add_argument('--rnn_hidden', type = int, default = 256)
parser.add_argument('--batch_size', type = int, default = 32)

opt = parser.parse_args()

def read_sentence(file):
    f = open(file, "r", encoding="utf-8")
    df = pd.read_csv(file, delimiter="\t", header = None)
    df.columns = ["label", "text"]
#     df = df.sample(frac=1.0, random_state=42)
    sentences = []
    labels = []
    for index, row in df.iterrows():
        label = int(row['label'])
        content = row['text']
        labels.append(label)
        sentences.append(content)
    print("{}:{}".format(file, len(sentences)))
    return sentences, labels

def read_sentence2(file):
    # For Qcorp
    f = open(file, "r", encoding="utf-8")
    df = pd.read_csv(file, delimiter="\t", header = None)
    df.columns = ["ID", "label", "text"]
#     df = df.sample(frac=1.0, random_state=42)
    sentences = []
    labels = []
    for index, row in df.iterrows():
        label = eval(row['label'])
        content = row['text']
        labels.append(label)
        sentences.append(content)
    print("{}:{}".format(file, len(sentences)))
    return sentences, labels

def read_file(folder, aug_num):
    sentences = []
    labels = []
    if folder == "Origindata/":
        f_train = f'{opt.dataset}/'+folder+"train.txt"
        f_dev = f'{opt.dataset}/'+folder+"dev.txt"
    else:
        f_train = f'{opt.dataset}/'+folder+f"train_{aug_num}.txt"
        
    f_dev = f'{opt.dataset}/Origindata/dev.txt' 
    f_test = f'{opt.dataset}/Origindata/test.txt'
    for file in [f_train, f_dev, f_test]:
        if opt.dataset != 'Qcorp':
            s, l = read_sentence(file)
        else:
            s, l = read_sentence2(file)
        sentences.extend(s)
        labels.extend(l)

    return sentences, labels

def write_json(file, content):
    with open(file, 'w') as file_obj:
        json.dump(content, file_obj)
    print("finish writing {}".format(file))

def parse_sentence(folder, sentences, labels, aug_num):
    if folder == "Origindata/":
        f_label = f'{opt.dataset}/' + folder + 'label.json'
        f_seg = f'{opt.dataset}/' + folder + 'seg.json'
        f_pos = f'{opt.dataset}/' + folder + 'pos.json'
        f_dep = f'{opt.dataset}/' + folder + 'dep.json'
    else:
        f_label = f'{opt.dataset}/' + folder + f'label_{aug_num}.json'
        f_seg = f'{opt.dataset}/' + folder + f'seg_{aug_num}.json'
        f_pos = f'{opt.dataset}/' + folder + f'pos_{aug_num}.json'
        f_dep = f'{opt.dataset}/' + folder + f'dep_{aug_num}.json'
    start = time.time()
    write_json(f_label, labels)
    print("finish writing labels, spending {} s.".format(time.time() - start))

    segs, poss, deps = [], [], []
    for i in tqdm(range(len(sentences))):
        seg, hidden = ltp_exm.seg([sentences[i]])
        pos = ltp_exm.pos(hidden)
        dep = ltp_exm.dep(hidden)
        segs.append(seg[0])
        poss.append(pos[0])
        deps.append(dep[0])

    write_json(f_seg, segs)
    write_json(f_pos, poss)
    write_json(f_dep, deps)
    print("finish parsing, spending {} s.".format(time.time() - start))

def concat_embed(folder, aug_num):
    if folder == "Origindata/":
        embed_train = np.load(f"{opt.dataset}/" + folder + f"{opt.model}_train_1_dim_256_batch_32.npy")
        embed_dev = np.load(f"{opt.dataset}/" + folder + f"{opt.model}_dev_1_dim_256_batch_32.npy")
        embed_test = np.load(f"{opt.dataset}/" + folder + f"{opt.model}_test_1_dim_256_batch_32.npy")
        embeds = [embed_train, embed_dev, embed_test]
        embeds = np.concatenate(embeds, axis=0)
        print(embeds.shape)
        np.save(f"{opt.dataset}/" + folder + f"{opt.model}_dim_256_batch_32_{embed_train.shape[0]}_{embed_dev.shape[0]}_{embed_test.shape[0]}.npy", embeds)
        os.remove(f"{root_file}/{opt.dataset}/" + folder + f"{opt.model}_train_1_dim_256_batch_32.npy")
        os.remove(f"{root_file}/{opt.dataset}/" + folder + f"{opt.model}_dev_1_dim_256_batch_32.npy")
        os.remove(f"{root_file}/{opt.dataset}/" + folder + f"{opt.model}_test_1_dim_256_batch_32.npy")
    else:
        embed_train = np.load(f"{opt.dataset}/" + folder + f"{opt.model}_train_{aug_num}_dim_256_batch_32.npy")
        embed_dev = np.load(f"{opt.dataset}/" + folder + f"{opt.model}_dev_{aug_num}_dim_256_batch_32.npy")
        embed_test = np.load(f"{opt.dataset}/" + folder + f"{opt.model}_test_{aug_num}_dim_256_batch_32.npy")
        embeds = [embed_train, embed_dev, embed_test]
        embeds = np.concatenate(embeds, axis=0)
        print(embeds.shape)
        np.save(f"{opt.dataset}/" + folder + f"{opt.model}_aug_{aug_num}_dim_256_batch_32_{embed_train.shape[0]}_{embed_dev.shape[0]}_{embed_test.shape[0]}.npy", embeds)
        
        os.remove(f"{root_file}/{opt.dataset}/" + folder + f"{opt.model}_train_{aug_num}_dim_256_batch_32.npy")
        os.remove(f"{root_file}/{opt.dataset}/" + folder + f"{opt.model}_dev_{aug_num}_dim_256_batch_32.npy")
        os.remove(f"{root_file}/{opt.dataset}/" + folder + f"{opt.model}_test_{aug_num}_dim_256_batch_32.npy")
    
    

if __name__ == '__main__':
    folder = opt.folder
    if folder ==  "Origindata/":
        aug_num = 0
        concat_embed(folder, aug_num)
        sentences, labels = read_file(folder, aug_num)
        parse_sentence(folder, sentences, labels, aug_num)
    else:
        if folder == "MKdata":
            aug_num = 3
            concat_embed(folder, aug_num)
            sentences, labels = read_file(folder, aug_num)
            parse_sentence(folder, sentences, labels, aug_num)
