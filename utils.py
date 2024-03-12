# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    # 创建词汇表
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        # 表示你正在遍历文件 f 中的每一行，并在循环中使用 tqdm 创建一个进度条来显示遍历的进度
        for line in tqdm(f):
            # lin为去一行文本首尾的空白字符的一行字符串
            lin = line.strip()
            # lin为空则跳过本次，进入下一轮，文章中间某一行可能有空字符串
            if not lin:
                continue
            # .split()方法，对字符串以制表符进行拆分，返回一个列表
            # train.txt内容：两天价网站背后重重迷雾：做个网站究竟要多少钱	4
            """
            .split将字符串以某种符合进行分隔并返回一个列表，使用何种分隔符号要具体问题具体分析
            content = lin.split('\t')
             content 现在是一个列表
             content = ["两天价网站背后重重迷雾：做个网站究竟要多少钱", "4"]"""
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                # 更新词汇表中词汇的出现次数，获取单词出现的次数：vocab_dic.get(word, 0)，未找到则返回0
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # [item for item in vocab_dic.items()] 将 vocab_dic 的键-值对转化为一个列表，
        # 例如 [('word1', 5), ('word2', 3), ('word3', 7), ...]
        # x 指的是待排列列表中的元素
        # 列表中的每个元素都是一个包含键和值的元组，而不包含冒号。
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        # 将排序后的词汇表中的每个词汇映射到一个索引，但是vocab_dic中的每个词丢失了频率信息
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level 对英文
    else:
        tokenizer = lambda x: [y for y in x]  # char-level  对中文
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 保存词汇表，避免重新构建
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        # pad_size 是指定填充后的序列长度
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                # 制表符和空格并不相同
                content, label = lin.split('\t')
                words_line = []
                # 将content拆分成列表
                token = tokenizer(content)
                # 获取序列长度
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        # 当序列长度小于pad_size 时，对序列进行填充，填充为PAD
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        # 当序列长度大于等于pad_size时，对序列进行截断
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    # words_line中存放的是token中各个词库在词汇表上对应的ID，找不到则存放的是UNK的ID
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        # contents是一个元素是三元组的列表
        # 每个元组中的第一个值是单词索引序列，第二个值是标签，第三个值是文本的序列长度。
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # batches是一个列表
        """
        batches = [
                        # 第一个样本
                        ([1, 4, 3, 7, 2], 0, 5),  # (文本数据的单词索引序列, 标签, 序列长度)
                        # 第二个样本
                        ([9, 6, 8, 0, 2], 1, 4),  # (文本数据的单词索引序列, 标签, 序列长度)
                        # 可能还有更多样本...
                    ]
        """
        # 这里的//不是注释，是整除，/代表浮点数除法
        self.n_batches = len(batches) // batch_size
        # 记录batch数量是否为整数，如果批次数量是整数，self.residue 会保持为 False
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # x为特征矩阵，y为标签,首先从数据中取出，然后再转化为tensor数据类型，才能输入当神经网络中
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        # 使用 seq_len 来告诉模型每个样本的实际长度，以便进行合适的填充
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            # 处理 有剩余数据不够组成一个batch_size的情况
            # 此处的:为切片操作，self.index * self.batch_size计算当前batch起始位置的索引
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
            # 准备输入下一个epoch的数据
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration  # 过时 直接return
        else:
            # 取index = 0时，会进入此条件，可知：该语句是处理每个batch_size数据的
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self
    # 此处我们定义的是迭代器，而不是数据库本身，故len函数所求的并不是样本数，而是迭代次数
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


"""
if __name__ == "__main__": 是一种用于区分模块作为主程序运行和模块被导入其他模块时执行不同代码的常见技巧。
如果一个Python脚本包含 if __name__ == "__main__": 代码块，那么这部分代码只会在该脚本作为主程序运行时执行，
而不会在其他模块导入它后自动调用
"""
if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
