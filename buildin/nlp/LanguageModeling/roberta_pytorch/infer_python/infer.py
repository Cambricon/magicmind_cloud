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
sys.path.append("../../../")
from utils.utils import Record

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
    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        assert args.device_id < dev_count
        # 打开MLU设备
        dev = mm.Device()
        dev.id = args.device_id
        assert dev.active().ok()
        # 创建Engine
        econfig = mm.Model.EngineConfig()
        econfig.device_type = "MLU"
        engine = model.create_i_engine(econfig)
        assert engine != None, "Failed to create engine"
        # 创建Context
        context = engine.create_i_context()
        assert context != None
        # 创建MLU任务队列
        queue = dev.create_queue()
        assert queue != None
        
        # 创建输入tensor
        inputs = context.create_inputs()
        
        tokenizer, eval_dataloader = preprocess(args.batch_size, args.max_seq_length)
        all_results = []
        all_labels = []
        epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
        for idx, batch in enumerate(epoch_iterator):
            text_batchs, text_labels = batch['text'], batch['label']
            encoding = tokenizer(text_batchs, return_tensors='np', padding="max_length", truncation=True,max_length=args.max_seq_length)
            assert inputs[0].from_numpy(encoding['input_ids']).ok()
            assert inputs[1].from_numpy(encoding['attention_mask']).ok()
            assert inputs[2].from_numpy(encoding['token_type_ids']).ok()
            outputs = []
            status = context.enqueue(inputs, outputs, queue)
            assert status.ok(), str(status)
            # 等待任务执行完成
            status = queue.sync()
            assert status.ok(), str(status)
            # 处理输出数据

            scores = outputs[0].asnumpy()
            senti_value = np.argmax(scores, axis=1)
            all_labels.extend(text_labels.numpy())
            all_results.extend(senti_value)
    ### 精度计算
    if args.compute_accuracy:

        acc_result = Record(args.acc_result)
        acc = accuracy_score(all_labels,all_results)
        acc_result.write("Accuracy results: {}".format(acc), True)
