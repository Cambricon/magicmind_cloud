import torch
import os
import transformers
from transformers import BertModel
import argparse
import warnings
from transformers import AutoTokenizer, BertForSequenceClassification
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "--model_path", type=str, default="../data/models/chinese-roberta-wwm-ext", help="model_path")
parser.add_argument("--onnx_model", "--onnx_model", type=str, default="../data/models/roberta.onnx", help="robert onnx")
parser.add_argument("--batch_size", "--batch_size", type=int, default=16, help="batch_size")
parser.add_argument("--max_seq_length", "--max_seq_length", type=int, default=128, help="max_seq_length")

if __name__ == "__main__":
    args = parser.parse_args()
    pt_model = BertForSequenceClassification.from_pretrained(args.model_path, from_tf = False, config = os.path.join(args.model_path, "config.json"), num_labels=2, cache_dir = None)
    pt_model.eval()

    tokens = torch.randint(0, 1, (args.batch_size, args.max_seq_length))
    segments = torch.randint(0, 1, (args.batch_size, args.max_seq_length))
    mask = torch.randint(0, 1, (args.batch_size, args.max_seq_length))

    # 保存模型文件
    input_names = ['input_ids','attention_mask','token_type_ids']
    output_names = ["output"]
    torch.onnx.export(pt_model, (tokens, segments, mask), args.onnx_model, verbose=True,opset_version=11,input_names=input_names,output_names=output_names) 
    print("successfully save onnx")
