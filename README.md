# 基于多图神经网络的领域知识和语法结构融合的中文医疗问询意图识别方法

## 数据
包括2个数据集：
- CMID: https://github.com/liutongyang/CMID
- Qcorp: https://www.heywhale.com/home/competition/5f2d0ea1b4ac2e002c164d82
(注：Qcorp相关比赛界面已关闭数据下载链接，故本项目只提供处理后的CMID数据)


使用的医学知识实体来自于 https://github.com/chenjj9527/QABasedOnMedicaKnowledgeGraph


## 运行代码

### 1. 融合知识的文本增强 (MKDA)

`cd ./MKDA`


`python augment.py --dataname CMID --input train.txt --num_aug 3 --msda 0 --mkda 1`


在CMID/MKdata 文件夹内已生成增强数据 train_3.txt，可不运行此步直接跳入下一步


### 2. 预训练语言模型微调

- ERNIE 1.0 预训练模型下载：https://github.com/nghuyong/ERNIE-Pytorch

- BERT-wwm：https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD

下载后在当前文件夹创建`ERNIE_pretrain`和`BERT_wwm_pretrain`文件夹，分别存放相关文件


`python CMID_bert_embedding.py --bert_path ERNIE_pretrain --data_type MKdata --train_file train`


### 3. 多图神经网络模型 (MGCN)

- 准备数据

`python generate_data.py --model ERNIE --dataset CMID --folder MKdata/`


- MGCN

`python GCN_concat_multiclass.py --dataset CMID --data_type MKdata --model ERNIE --beta 0.01 --aug_num 3 --times 1`


## 运行环境
本实验在 python 3.8, pytorch 1.8 环境中进行

具体的依赖项可参考 `requirements.txt`

## 实验结果

### CMID数据集 （5次结果平均）

|               | Precision (%)| Recall (%)  | F1  (%)      |
| ------------- | ------------ | ------------ | ------------ |
| BERT_wwm      | 69.56(±0.71) | 71.29(±0.75) | 69.44(±0.61) |
| + MKDA        | 70.71(±0.74) | 72.04(±0.75) | 70.25(±0.69) |
| + MGCN        | 70.11(±0.56) | 70.82(±0.63) | 70.43(±0.47) |
| + MKDA + MGCN | 71.09(±0.80) | 72.34(±0.82) | 71.09(±0.66) |
| ERNIE         | 71.20(±0.73) | 72.48(±1.01) | 71.28(±0.67) |
| + MKDA        | 72.05(±0.27) | 73.52(±0.35) | 71.91(±0.35) |
| + MGCN        | 72.44(±0.64) | 73.50(±0.66) | 72.41(±0.39) |
| + MKDA + MGCN | 73.13(±0.58) | 73.98(±0.35) | 73.16(±0.53) |


