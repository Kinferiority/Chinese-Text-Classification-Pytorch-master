import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
from transformers import BertModel
from transformers import BertTokenizer

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
        label = int(self.df['Label'][idx])
        return text,label
#定义下游任务模型
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768,10)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out




device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model()

# 加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')

# 移动到设备上
pretrained.to(device)
# 不训练,不需要计算梯度
# for param in pretrained.parameters():
#     param.requires_grad_(False)
token = BertTokenizer.from_pretrained('bert-base-chinese')
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels
train_path = "./THUCNews/data/train.txt"
train_data = myDataset(train_path)
loader = torch.utils.data.DataLoader(dataset=train_data,
                                     batch_size=512,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
from transformers import AdamW

#训练
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)

        print(i, loss.item(), accuracy)

    if i == 100:
        break
val_file = "./THUCNews/data/dev.txt"
val_dataset = myDataset(val_file)
def test():
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=200,
                                              collate_fn=collate_fn,
                                              shuffle=False,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):

        if i == 5:
            break

        print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)


test()