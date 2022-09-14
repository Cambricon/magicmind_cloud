import argparse
import magicmind.python.runtime as mm
import torch
import os
from tqdm import tqdm
import transformers
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
import warnings
warnings.filterwarnings("ignore")
from preprocess import preprocess
from postprocess import postprocess
import sys
sys.path.append("../../../")
from utils.utils import Record

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id", type = int, default = 0, help = "device_id")
parser.add_argument("--magicmind_model", "--magicmind_model", type = str, default = "../data/models/bert_qa_pytorch_model")
parser.add_argument("--json_file", "--json_file", type = str, default = "../../../../../../datasets/squadv1.1/dev-v1.1.json")
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
        
        tokenizer, examples, features, eval_dataloader = preprocess(args.json_file, args.batch_size, args.max_seq_length)
        all_results = []
        for batch in tqdm(eval_dataloader, desc = "Evaluating"):
            batch = tuple(t for t in batch)
            # 准备输入数据
            for i in range(3):
                inputs[i].from_numpy(batch[i].numpy())
            # 向MLU下发任务
            outputs = []
            status = context.enqueue(inputs, outputs, queue)
            assert status.ok(), str(status)
            # 等待任务执行完成
            status = queue.sync()
            assert status.ok(), str(status)
            # 处理输出数据
            outputs_np = []
            for tensor in outputs:
                outputs_np.append(tensor.asnumpy())
            # import pdb;pdb.set_trace()
            all_results.extend(postprocess(features, batch[3], outputs_np))

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
            tokenizer
        )
        squad_acc = squad_evaluate(examples, predictions)
        acc_result.write("SQUAD results: {}".format(squad_acc), True)
