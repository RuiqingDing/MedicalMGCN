# @Author : zhany
# @Time : 2019/03/20 

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mseda import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--dataname", required=True, type=str, help='数据集名称')
ap.add_argument("--input", required=True, type=str, help="原始数据的输入文件目录")
ap.add_argument("--output", required=False, type=str, help="增强数据后的输出文件目录")
ap.add_argument("--num_aug", required=True, type=int, help="每条原始语句增强的语句数")
ap.add_argument("--alpha_sr",required=False,type=float,help="sr改变单词数占比")
ap.add_argument("--alpha_ri",required=False,type=float,help="ri改变单词数占比")
ap.add_argument("--alpha_rd",required=False,type=float,help="rd改变单词数占比")
ap.add_argument("--alpha_rs",required=False,type=float,help="rs改变单词数占比")
ap.add_argument("--msda",required=True, type=bool,help="是否加入替换专业术语的增强方式+EDA")
ap.add_argument("--mkda", required=True, type=bool, help='是否只用替换专业术语+同义词替换')
args = ap.parse_args()

#输入文件
input_file = f'../{args.dataname}/Origindata/'+args.input

#每条原始语句增强的语句数
num_aug = 0 #default
if args.num_aug:
    num_aug = args.num_aug

# 是否增加medical specific augment
msda = False #default
if args.msda:
    msda = args.msda

#每条语句中将会被改变的单词数占比
alpha_sr = 0.1#default
if args.alpha_sr:
    alpha_sr = args.alpha_sr

#每条语句中将会被改变的单词数占比
alpha_ri = 0.1#default
if args.alpha_ri:
    alpha_sr = args.alpha_ri
    
#每条语句中将会被改变的单词数占比
alpha_rd = 0.1#default
if args.alpha_rd:
    alpha_sr = args.alpha_rd
    
#每条语句中将会被改变的单词数占比
alpha_rs = 0.1#default
if args.alpha_rs:
    alpha_sr = args.alpha_rs

dict_medical_category_pick = {}
dict_medical_pick = {}
if args.mkda:
    # 医学相关的各类词语
    print('生成和本数据集相关的医学字典，避免引入太多复杂的术语')
    f = open('dict_medical.txt', 'r', encoding='utf-8')
    dict_medical = eval(f.read())
    f.close()

    dict_medical['symptom'].remove('医生')

    dict_medical_category = {}
    for key in dict_medical:
        terms = dict_medical[key]
        for term in terms:
            if term not in dict_medical_category:
                dict_medical_category[term] = key
    
    dict_medical_category_pick = {}
    dict_medical_pick = {}
    lines = open(input_file, 'r', encoding='utf-8').readlines()
    for line in lines:
        parts = line[:-1].split('\t')    #使用[:-1]是把\n去掉了
        if len(parts) == 2:
            label = parts[0]
            sentence = parts[1]
        else:
            label = parts[1]
            sentence = parts[2]
        seg_list = jieba.cut(sentence)
        seg_list = " ".join(seg_list)
        words = list(seg_list.split())
        for word in words:
            if word in dict_medical_category:
                if word not in dict_medical_category_pick:
                    dict_medical_category_pick[word] = dict_medical_category[word]
                    if dict_medical_category[word] not in dict_medical_pick:
                        dict_medical_pick[dict_medical_category[word]] = [word]
                    else:
                        dict_medical_pick[dict_medical_category[word]].append(word)
                    
                    
    print('len(dict_medical_category_pick) = ', len(dict_medical_category_pick))
                
    
    
def gen_mseda(input_file, output_file, alpha_sr, alpha_ri, alpha_rs,p_rd, num_aug, msda, mkda):
    writer = open(output_file, 'w', encoding='utf-8')
    lines = open(input_file, 'r', encoding='utf-8').readlines()
    
    cnt=0
    
    print("正在使用MSEDA生成增强语句...")
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')    #使用[:-1]是把\n去掉了
        if len(parts) == 2:
            label = parts[0]
            sentence = parts[1]
            aug_sentences = mseda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd,
                                  num_aug=num_aug, msda=msda, mkda=mkda, dict_medical = dict_medical_pick, dict_medical_category = dict_medical_category_pick)
            for aug_sentence in aug_sentences:
                writer.write(label + "\t" + "".join(aug_sentence.split(' ')) + '\n')
        else:
            id = parts[0]
            label = parts[1]
            sentence = parts[2]
            aug_sentences = mseda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd,
                                  num_aug=num_aug, msda=msda, mkda=mkda, dict_medical = dict_medical_pick, dict_medical_category = dict_medical_category_pick)
            for aug_sentence in aug_sentences:
                writer.write(id + "\t" + label + "\t" + "".join(aug_sentence.split(' ')) + '\n')

        cnt+=1
        if cnt % 10 == 0:
            print("正在生成增强语句：第"+str(cnt)+'个')

    writer.close()
    print("已生成增强语句!")
    print(output_file)

if __name__ == "__main__":
    print(input_file)
    
    #输出文件
    if ((args.msda) & (not args.mkda)):
        output_file = f'../{args.dataname}/Alldata/'+args.input.split('.')[0]+f'_{args.num_aug}.txt'
    else:
        output_file = f'../{args.dataname}/MKdata/'+args.input.split('.')[0]+f'_{args.num_aug}.txt'
        
    print(output_file)
    gen_mseda(input_file, output_file,alpha_sr=alpha_sr, alpha_ri=alpha_ri,
              alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug, msda=args.msda, mkda=args.mkda)