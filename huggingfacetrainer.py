from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
from datasets import load_from_disk, load_dataset

#加载分词工具
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def txt_to_pd(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.readlines()  # 一个包含文件中所有行的列表：['Hello\n', 'World\n', 'How\n', 'Are\n', 'You\n']
    text_list = []
    label_list = []
    for line in data:
        # .split('\t') 以该符号为间隔，返回列表
        line = line.strip().split('\t')  # 移除字符串开头和结尾的空白字符（例如空格、制表符、换行符等）的方法
        text_list.append(line[0])
        label_list.append(line[1])
    df = pd.DataFrame({
        'Text': text_list,
        'Label': label_list
    })
    return df
class myDataset(Dataset):
    def __init__(self,filename):
        self.df = txt_to_pd(filename)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        text = self.df['Text'][idx]
        label = self.df['Label'][idx]
        return text,label

def f(data):
    return tokenizer(
        data['Text'],
        padding='max_length',
        truncation=True,
        max_length=30,
    )

from transformers import AutoModelForSequenceClassification
train_path = "./THUCNews/data/train.txt"
train_data = load_dataset("text", data_files=train_path)
train_data=train_data.map(f, batched=True, batch_size=1000, num_proc=4)
#加载模型
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased',
                                                           num_labels=10)