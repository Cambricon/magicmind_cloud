import argparse
import magicmind.python.runtime as mm
import torch
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import transformers
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")
from preprocess import preprocess
import sys
from mm_runner import MMRunner
from logger import Logger
from utils import Record
logging = Logger()

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id", type = int, default = 0, help = "device_id")
parser.add_argument("--magicmind_model", "--magicmind_model", type = str, default = "../data/models/bert_qa_pytorch_model")
parser.add_argument("--batch_size", "--batch_size", type = int, default = 16, help = "batch_size")
parser.add_argument("--max_seq_length", "--max_seq_length", type = int, default = 128, help = "max_seq_length")
parser.add_argument("--compute_accuracy", "--compute_accuracy", type = bool, default = True)
parser.add_argument("--output_dir", "--output_dir", type = str, default = "../data/jsons/output")
parser.add_argument("--acc_result", "--acc_result", type = str, default = "../data/jsons/output/acc_result.txt")


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    model = MMRunner(mm_file=args.magicmind_model, device_id=args.device_id)

    tokenizer, eval_dataloader = preprocess(args.batch_size, args.max_seq_length)
    all_results = []
    all_labels = []
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    for idx, batch in enumerate(epoch_iterator):
        text_batchs, text_labels = batch['text'], batch['label']
        encoding = tokenizer(text_batchs, return_tensors='np', padding="max_length", truncation=True,max_length=args.max_seq_length)
        inputs = [encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids']]
        # 处理输出数据
        scores = model(inputs)[0]
        senti_value = np.argmax(scores, axis=1)
        all_labels.extend(text_labels.numpy())
        all_results.extend(senti_value)
    ### 精度计算
    if args.compute_accuracy:

        acc_result = Record(args.acc_result)
        acc = accuracy_score(all_labels,all_results)
        acc_result.write("Accuracy results: {}".format(acc), True)
