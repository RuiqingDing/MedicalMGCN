# 基于多图神经网络的领域知识和语法结构融合的中文医疗问询意图识别方法

## 数据
包括2个数据集：
- CMID: https://github.com/liutongyang/CMID
- Qcorp: https://www.heywhale.com/home/competition/5f2d0ea1b4ac2e002c164d82
(注：Qcorp相关比赛界面已关闭数据下载链接，故本项目只提供处理后的CMID数据)

使用的医学知识实体来自于 https://github.com/chenjj9527/QABasedOnMedicaKnowledgeGraph


## 运行代码

### 融合知识的文本增强

`cd ./MKDA
python augment.py --dataname CMID --input train.txt --num_aug 3 --msda 0 --mkda 1`

在CMID/MKdata 文件夹内已生成增强数据 train_3.txt，可不运行此步直接跳入下一步

### 预训练语言模型微调
`cd ./`
`python CMID_bert_embedding.py --bert_path ERNIE_pretrain --data_type MKdata --train_file train`


### 多图神经网络模型

1. 准备数据
`python generate_data.py --model ERNIE --dataset CMID --folder MKdata/`


2、MGCN
`python GCN_concat_multiclass.py --dataset CMID --data_type MKdata --model ERNIE --beta 0.01 --aug_num 3 --times 1`


## 运行环境
本实验在 python 3.8, pytorch 1.8 环境中进行
具体的依赖项可参考 `requirements.txt`



