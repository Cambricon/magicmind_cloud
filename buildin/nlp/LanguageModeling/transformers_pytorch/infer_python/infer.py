import magicmind.python.runtime as mm
import argparse
import numpy as np
from tqdm import tqdm
import time
from transformers import AutoTokenizer,default_data_collator
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_pt_utils import nested_concat,nested_truncate
from torch.utils.data import DataLoader, SequentialSampler
from datasets import load_dataset, load_metric

import os 

cur_dir = os.path.dirname(os.path.abspath(__file__))

# metrics
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result
    
# data preprocess
def preprocess_function(examples):
    result = tokenizer(examples['sentence1'], examples['sentence2'],padding='max_length', max_length=MAX_SEQ_LENGTH, truncation=True)
    return result

parser = argparse.ArgumentParser(description='mAP Calculation')
parser.add_argument('--magicmind_model', type=str, required=True)
parser.add_argument('--dev_id', help='The result data path', type=int)
parser.add_argument('--batch_size', help='batch_size', type=int, default=1)
parser.add_argument('--datasets_dir', type=str, default="")
parser.add_argument('--acc_result', help='acc_result', type=str, default="")
parser.add_argument('--test_nums', help='test_nums', type=int, default=-1)

if __name__ == "__main__":
    args = parser.parse_args()
    
    DEV_ID = args.dev_id
    MM_MODEL = args.magicmind_model
    MODEL_NAME = 'textattack/bert-base-uncased-MRPC'
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LENGTH = 128
    
    # load glue mrpc dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_datasets = load_dataset('glue', 'mrpc', cache_dir=args.datasets_dir)
    raw_datasets = raw_datasets.map(preprocess_function, batched=True,
                                    load_from_cache_file=False,keep_in_memory=True,desc="Running tokenizer on dataset")
    eval_dataset = raw_datasets['validation']
    eval_sampler = SequentialSampler(eval_dataset)
    dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=BATCH_SIZE,
                            collate_fn=default_data_collator, drop_last=True)

    metric = load_metric(os.path.join(cur_dir,'../export_model/glue.py'), 'mrpc')
        
    with mm.System() as mm_sys:  # 初始化系统
        dev_count = mm_sys.device_count()
        assert DEV_ID < dev_count
        # 打开MLU设备
        dev = mm.Device()
        dev.id = DEV_ID
        assert dev.active().ok()
        # 反序列化离线模型
        model = mm.Model()
        model.deserialize_from_file(args.magicmind_model)
        engine = model.create_i_engine()
        assert engine is not None
        # 创建Context
        context = engine.create_i_context()
        assert context is not None
        # 创建MLU任务队列
        queue = dev.create_queue()
        assert queue is not None
        # 创建输入tensor
        input_tensors = context.create_inputs()

        all_preds = None
        all_labels = None
        start_time = time.time()
        for step, inputs in tqdm(enumerate(dataloader), desc='Evaluating...'):
            if step > args.test_nums:break
            output_tensors = []
            input_tensors[0].from_numpy(inputs['input_ids'].numpy())
            input_tensors[1].from_numpy(inputs['attention_mask'].numpy())
            input_tensors[2].from_numpy(inputs['token_type_ids'].numpy())
            assert context.enqueue(input_tensors, output_tensors, queue).ok()
            assert queue.sync().ok()
            logits = output_tensors[0].asnumpy()
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits)
            all_labels = inputs['labels'] if all_labels is None else nested_concat(all_labels, inputs['labels'])
        end_time = time.time()
        raw_datasets.cleanup_cache_files() 
        # eval preds
        num_samples = len(eval_dataset)
        all_preds  = nested_truncate(all_preds, num_samples)
        all_labels = nested_truncate(all_labels, num_samples)
        metrics = compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        # prefix
        with open(args.acc_result,'w') as f:
            for key in list(metrics.keys()):
                if not key.startswith(f"eval_"):
                    metrics[f"{key}"] = metrics.pop(key)
                    f.writelines(key+":" + str(metrics[f"{key}"]) + '\n')
        f.close()
        throughput = num_samples / (end_time - start_time)
        print(metrics)
        print('throughput: %f samples per second' % throughput)
