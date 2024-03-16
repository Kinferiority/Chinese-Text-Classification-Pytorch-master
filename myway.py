import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from gensim.models import KeyedVectors
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'TextCNN'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.dropout = 0.5  # 随机失活
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.num_classes = 10


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


df_train = txt_to_pd("./THUCNews/data/train.txt")
df_test = txt_to_pd("./THUCNews/data/test.txt")
df_dev = txt_to_pd("./THUCNews/data/dev.txt")
# print(df_train.head())


word2vec_model_path = '/Users/jrk/Downloads/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)


def tokenize_and_vectorize(sentence, max_length=32):
    tokens = jieba.lcut(sentence)
    features = []
    for token in tokens[:max_length]:
        if token in word2vec_model:
            features.append(word2vec_model[token])
        else:
            features.append(np.random.rand(word2vec_model.vector_size))
    while len(features) < max_length:
        features.append(np.zeros(word2vec_model.vector_size))
    return np.array(features)


class TextData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        text = tokenize_and_vectorize(self.data['Text'][idx])
        label = int(self.data['Label'][idx])
        return text, label

    def __len__(self):
        return len(self.data['Text'])


train_dataset = TextData(df_train)
print(train_dataset[0][0].shape)
train_loader = DataLoader(dataset=train_dataset, batch_size=Config().batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, 64)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # 输入x的维度：(batch_size, sequence_length)
        # 实例化 embedding
        # 将文本数据视为一个通道的图像数据,在第二维（channel维度）上添加一个新维度
        # (batch_size, 1, sequence_length, embedding_dim)
        out = x.float()
        out = out.unsqueeze(1)
        # 将三个经过池化后的矩阵横着拼接
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


model = CNN(Config()).to(device=Config().device)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr)

for j in range(Config().num_epochs):
    loss_all = []
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        text_embed = data[0].to(Config().device)
        labels = data[1]
        output = model(text_embed)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        loss_all.append(loss)
    print(f"第{j + 1}轮的损失为{loss_all[j]}")

