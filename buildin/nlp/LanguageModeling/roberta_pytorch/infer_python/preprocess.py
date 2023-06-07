import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import BertTokenizer
import csv
from itertools import islice
import os

class MyDataSet(Dataset):
    def __init__(self, corpus_file):
        self.data = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for item in islice(reader, 1, None): # 忽略表头
                item = item[0].split("\t")
                label, text = item[0], item[1]
                self.data.append((text, label))

    def __getitem__(self, index):
        return {"text": self.data[index][0], "label": int(self.data[index][1])}

    def __len__(self):
        return len(self.data)

def preprocess(batch_size, max_seq_length):
    ###preprocess data
    model_path = os.environ.get('MODEL_PATH')+ "/chinese-roberta-wwm-ext-chnsenticorp"
    dataset_path = os.environ.get('CHNSENTICORP_DATASETS_PATH')+ "/test.tsv"
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case = False)
    dataset = MyDataSet(dataset_path)
    eval_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    print("Num examples = ", len(dataset))
    print("Batch size = ", batch_size)
    print("Iterations = ", len(eval_dataloader))
    return tokenizer, eval_dataloader
