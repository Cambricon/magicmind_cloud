import torch
import os
import transformers
from transformers import AutoModelForQuestionAnswering
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "--model_path", type=str, default="../data/models/pytorch_bert_base_cased_squad", help="model_path")
parser.add_argument("--pt_model", "--pt_model", type=str, default="../data/models/bert_squad_pytorch.pt", help="bert_squad_pytorch pt")
parser.add_argument("--batch_size", "--batch_size", type=int, default=16, help="batch_size")
parser.add_argument("--max_seq_length", "--max_seq_length", type=int, default=128, help="max_seq_length")

if __name__ == "__main__":
    args = parser.parse_args()
 
    pt_model = AutoModelForQuestionAnswering.from_pretrained(args.model_path, from_tf = False, config = None, cache_dir = None)
    pt_model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location = 'cpu'))
    pt_model.eval()

    tokens = torch.randint(0, 1, (args.batch_size, args.max_seq_length))
    segments = torch.randint(0, 1, (args.batch_size, args.max_seq_length))
    mask = torch.randint(0, 1, (args.batch_size, args.max_seq_length))

    # 保存模型文件
    torch.jit.save(torch.jit.trace(pt_model, (tokens, segments, mask)), args.pt_model)
    print("successfully save pt")
