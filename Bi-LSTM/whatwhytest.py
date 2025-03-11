
import torch
from datasets import load_dataset  # hugging-face dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import os
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_float32_matmul_precision('high')
PATH = "./lightning_logs/version_1/checkpoints/epoch=29-step=330.ckpt"
batch_size = 64
epochs = 30
dropout = 0.4
rnn_hidden = 768
rnn_layer = 1
class_num = 4
lr = 0.001

# todo：自定义数据集
import json

class MydataSet(Dataset):
    def __init__(self, path, split):
        self.data = []
        with open('go_output.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                self.data.append(json.loads(line))
        # 根据split参数（例如“train”、“test”等）进一步处理self.data

    def __getitem__(self, item):
        text = self.data[item]['msg']  # 根据实际的键名调整
        label = 0  # 根据实际的键名调整
        return text, label

    def __len__(self):
        return len(self.data)


# todo: 定义批处理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=200,
        return_tensors='pt',  # 以pytorch的形式返回，可取值tf,pt,np,默认为返回list
        return_length=True,
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']  # input_ids 就是编码后的词
    attention_mask = data['attention_mask']  # pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # (如果是一对句子)第一个句子和特殊符号的位置是0,第二个句子的位置是1
    labels = torch.LongTensor(labels)  # 该批次的labels

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# todo: 定义模型，上游使用bert预训练，下游任务选择双向LSTM模型，最后加一个全连接层
class BiLSTMClassifier(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型,生成embedding层
        self.embedding = BertModel.from_pretrained('bert-base-uncased')
        # 去掉移至gpu
        # 冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # 生成下游RNN层以及全连接层
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=self.drop)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # 使用CrossEntropyLoss作为损失函数时，不需要激活。因为实际上CrossEntropyLoss将softmax-log-NLLLoss一并实现的。

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state  # 第0维才是我们需要的embedding,embedding.last_hidden_state = embedding[0]
        out, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.fc(output)
        return output


# todo: 定义pytorch lightning
class BiLSTMLighting(pl.LightningModule):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMLighting, self).__init__()
        self.model = BiLSTMClassifier(drop, hidden_dim, output_dim)  # 设置model
        self.criterion = nn.CrossEntropyLoss()  # 设置损失函数
        self.train_dataset = MydataSet('./data/archive/train_clean.csv', 'train')
        self.val_dataset = MydataSet('./data/archive/val_clean.csv', 'train')
        self.test_dataset = MydataSet('./data/archive/test_cleanjava.csv', 'train')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch  # x, y = batch
        y = one_hot(labels, num_classes=4)
        # 将one_hot_labels类型转换成float
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()  # 将[128, 1, 3]挤压为[128,3]
        loss = self.criterion(y_hat, y)  # criterion(input, target)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 将loss输出在控制台
        return loss  # 必须把log返回回去才有用

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return val_loader

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        y = one_hot(labels, num_classes=4)
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return test_loader

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        # 前向传播
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        # 预测
        pred = torch.argmax(y_hat, dim=1)
        with open("predgo.txt", "a", encoding="utf-8") as f:
            f.write(str(pred))
        # 返回预测结果和原始数据
        return {"pred": pred, "text": [i[0] for i in batch], "label": labels}

    def on_test_epoch_end(self):
        all_predictions = []
        for output in self.trainer.callback_metrics:
            preds = output["pred"].tolist()
            texts = output["text"]
            labels = output["label"].tolist()

            for pred, text, label in zip(preds, texts, labels):
                updated_record = {"new_message1": text, "label": label, "result": pred}
                all_predictions.append(updated_record)

        # 保存为新的 JSONL 文件
        with open("predictions.jsonl", "w", encoding="utf-8") as file:
            for record in all_predictions:
                file.write(json.dumps(record) + "\n")


def test():
    # 加载之前训练好的最优模型参数
    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=PATH,
                                                drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    trainer = Trainer(fast_dev_run=False)
    result = trainer.test(model)
    print(result)

token = BertTokenizer.from_pretrained('bert-base-uncased')
if __name__ == '__main__':

    model = BiLSTMLighting.load_from_checkpoint(checkpoint_path=PATH, drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    trainer = Trainer()
    trainer.test(model)