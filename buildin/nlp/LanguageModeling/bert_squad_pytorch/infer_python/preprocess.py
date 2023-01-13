import torch
from torch.utils.data import DataLoader, SequentialSampler
import transformers
from transformers import AutoTokenizer, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor
import os

def preprocess(json_file, batch_size, max_seq_length):
    ###preprocess data
    model_path = os.environ.get('MODEL_PATH')+ "/bert-squad-training"
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case = False)
    squad_processor = SquadV1Processor()
    examples = squad_processor.get_dev_examples("", filename=json_file)
    features, dataset = squad_convert_examples_to_features(
        examples = examples,
        tokenizer = tokenizer,
        max_seq_length = max_seq_length,
        doc_stride = 128,
        max_query_length = 64,
        is_training = False,
        return_dataset = "pt",
        threads = 4)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler = eval_sampler, batch_size = batch_size, drop_last = False)
    print("Num examples = ", len(dataset))
    print("Batch size = ", batch_size)
    print("Iterations = ", len(eval_dataloader))
    return tokenizer, examples, features, eval_dataloader
