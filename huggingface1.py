import torch
from torch.utils.data import Dataset

from datasets import load_dataset

data_files = {"train": "./THUCNews/data/train.csv", "test": "./THUCNews/data/test.csv","val":"./THUCNews/data/dev.csv"}
dataset = load_dataset('csv',data_files=data_files)
print(dataset)
print(dataset['train'][0])

train_data = dataset['train']
valid_data = dataset['val']
test_data = dataset['test']

print(test_data[0])

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
checkpoint ='bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def collote_fn(batch_samples):
    batch_text = [i['Text']for i in batch_samples]
    batch_label = [i['Label']for i in batch_samples]
    X = tokenizer(
        batch_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X,y


train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=512, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False, collate_fn=collote_fn)

batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)

from torch import nn
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, 10)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0]
        logits = self.classifier(cls_vectors)
        return logits

model = NeuralNetwork().to(device)
print(model)

from tqdm.auto import tqdm


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")
    return correct

from transformers import AdamW, get_scheduler

learning_rate = 1e-5
epoch_num = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_acc = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")