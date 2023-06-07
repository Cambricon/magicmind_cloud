import argparse
import magicmind.python.runtime as mm
import torch
import os
from tqdm import tqdm
import transformers
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
import warnings

warnings.filterwarnings("ignore")
from preprocess import preprocess
from postprocess import postprocess
import sys
import numpy as np

from utils import Record
import pickle

from mm_runner import MMRunner
from logger import Logger

log = Logger()

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id", type=int, default=0, help="device_id")
parser.add_argument(
    "--magicmind_model",
    "--magicmind_model",
    type=str,
    default="../data/models/bert_squad_pytorch_model",
)
parser.add_argument(
    "--json_file",
    "--json_file",
    type=str,
    default="../../../../../../datasets/squadv1.1/dev-v1.1.json",
)
parser.add_argument(
    "--batch_size", "--batch_size", type=int, default=16, help="batch_size"
)
parser.add_argument(
    "--max_seq_length", "--max_seq_length", type=int, default=384, help="max_seq_length"
)
parser.add_argument("--compute_accuracy", "--compute_accuracy", type=bool, default=True)
parser.add_argument(
    "--output_dir", "--output_dir", type=str, default="../data/jsons/output"
)
parser.add_argument(
    "--acc_result",
    "--acc_result",
    type=str,
    default="../data/jsons/output/acc_result.txt",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        log.info("please generate magicmind model first!!!")
        exit()

    # model 定义
    model = MMRunner(mm_file=args.magicmind_model, device_id=args.device_id)

    tokenizer, examples, features, eval_dataloader = preprocess(
        args.json_file, 1, args.max_seq_length
    )

    data_num = len(eval_dataloader)
    batch_size = args.batch_size
    max_seq_length = args.max_seq_length
    rem_data_num = data_num % args.batch_size
    data_idx = 0
    batch_counter = 0
    batch_input_1 = np.empty([batch_size, max_seq_length])
    batch_input_2 = np.empty([batch_size, max_seq_length])
    batch_input_3 = np.empty([batch_size, max_seq_length])
    post_batch = np.empty([batch_size, 1], dtype=np.int)
    all_results = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        infer_batch_size = (
            batch_size if data_idx < (data_num - rem_data_num) else rem_data_num
        )

        batch = tuple(t for t in batch)

        batch_input_1[batch_counter % infer_batch_size] = batch[0][0].numpy()
        batch_input_2[batch_counter % infer_batch_size] = batch[1][0].numpy()
        batch_input_3[batch_counter % infer_batch_size] = batch[2][0].numpy()
        post_batch[batch_counter % infer_batch_size] = batch[3][0].numpy()

        batch_counter += 1
        data_idx += 1

        if batch_counter % infer_batch_size == 0:
            batch_counter = 0
            inputs = [batch_input_1, batch_input_2, batch_input_3]

            # infer
            outputs = model(inputs)

            post_outputs = [
                outputs[0][0:infer_batch_size, :],
                outputs[1][0:infer_batch_size, :],
            ]

            # process output
            all_results.extend(
                postprocess(
                    features,
                    torch.from_numpy(post_batch[0:infer_batch_size, :]),
                    post_outputs,
                )
            )

    ### 精度计算
    if args.compute_accuracy:
        acc_result = Record(args.acc_result)
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            20,  # n best size
            30,  # max answer length
            False,  # do lower case
            output_prediction_file,
            output_nbest_file,
            None,
            False,
            False,
            0.0,
            tokenizer,
        )
        squad_acc = squad_evaluate(examples, predictions)
        acc_result.write("SQUAD results: {}".format(squad_acc), True)
