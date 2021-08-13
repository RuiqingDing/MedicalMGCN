#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jieba
import numpy as np
import synonyms
import random
from random import shuffle
from tqdm import tqdm
jieba.load_userdict("jiebaDict.txt") #add medical terms when cutting sentences

random.seed(2021)

#停用词列表，默认使用哈工大停用词表
f = open('HIT_stop_words.txt',encoding='utf-8')
stop_words = list()
for stop_word in f.readlines():
    stop_words.append(stop_word[:-1])


# # 医学相关的各类词语
# f = open('dict/dict_medical.txt', 'r', encoding='utf-8')
# dict_medical = eval(f.read())
# f.close()

# dict_medical['symptom'].remove('医生')

# dict_medical_category = {}
# for key in dict_medical:
#     terms = dict_medical[key]
#     for term in terms:
#         if term not in dict_medical_category:
#             dict_medical_category[term] = key

#考虑到与英文的不同，暂时搁置
#文本清理
'''
import re
def get_only_chars(line):
    #1.清除所有的数字
'''

########################################################################
# 同义词替换
# 替换一个语句中的n个单词为其同义词
########################################################################
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))     
    random.shuffle(random_word_list)
    num_replaced = 0  
    for random_word in random_word_list:          
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)   
            new_words = [synonym if word  == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: 
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    return synonyms.nearby(word)[0]


########################################################################
# 随机插入
# 随机在语句中插入n个词
########################################################################
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0    
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# 随机删除
# 以概率p删除语句中的词
########################################################################
def random_deletion(words, p):

    if len(words)*p <= 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words


########################################################################
# 专业词语替换
# 若句子中有专业名词出现，则将句子中的词语替换成同一类别的其他词语
########################################################################
def medical_term_replacement(words, dict_medical, dict_medical_category):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)

    term_sign = 0
    for word in random_word_list:
        if word in dict_medical:
            term_sign = 1
            key = dict_medical_category[word]
            candidates = [w for w in dict_medical[key] if w != word]
            if len(candidates) > 0:
                replacement = random.choice(candidates)
                new_words = [replacement if w == word else w for w in new_words]
            else:
                new_words = new_words
            break
    if term_sign == 1:
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words
    if term_sign == 0: #没有找到专业名词
        return synonym_replacement(words, min(np.ceil(0.1*len(words)), 10)) #最多替换句子中的5个词

########################################################################
#MSEDA函数
def mseda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=5, msda = True, mkda = True, dict_medical={},dict_medical_category={}):
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    if mkda:
        # 一半是专业词汇替换，一半是同义词替换
        for _ in range(int(num_aug/2)+1):
            a_words = medical_term_replacement(words, dict_medical, dict_medical_category)
            augmented_sentences.append(' '.join(a_words))
        for _ in range(int(num_aug/2)+1):
            n_sr = min(int(np.ceil(alpha_sr * num_words)), 5)
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))
    else:
        if msda:
            num_new_per_technique = int((num_aug-1)/4) + 1
            a_words = medical_term_replacement(words, dict_medical, dict_medical_category)
            augmented_sentences.append(' '.join(a_words))

        else:
            num_new_per_technique = int(num_aug/4) + 1

        n_sr = min(int(np.ceil(alpha_sr * num_words)), 5)
        n_ri = min(int(np.ceil(alpha_ri * num_words)), 5)
        n_rs = min(int(np.ceil(alpha_rs * num_words)), 5)

        #同义词替换sr
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

        #随机插入ri
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

        #随机交换rs
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

        #随机删除rd
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    shuffle(augmented_sentences)
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(seg_list)

    return augmented_sentences

##
#测试用例
#print(mseda(sentence="我们就像蒲公英，我也祈祷着能和你飞去同一片土地"))
# print(mseda(sentence='瘰疬性皮肤结核病是一种比较严重的疾病，可以去查一下皮肤科', num_aug=2))