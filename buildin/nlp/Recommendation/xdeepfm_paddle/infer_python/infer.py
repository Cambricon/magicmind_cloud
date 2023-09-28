import os
import numpy as np
import argparse
import sys
import time
from utils import Record
from mm_runner import MMRunner
from logger import Logger
log = Logger()
from metric import Auc
sys.path.append("..")
from gen_model.criteo_reader import RecDataset
import paddle
from paddle.io import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", "--device_id", type=int, default=0, help="device_id")
parser.add_argument("--magicmind_model","--magicmind_model",type=str,default="../data/mm_model/xdeepfm_onnx_model_force_float32_true_512", help="save mm model to this path")
parser.add_argument("--batch_size","--batch_size", type=int,default=512,help="batch_size used for infer")
parser.add_argument("--dataset_dir","--dataset_dir",type=str,default="/nfsdata/datasets/criteo/slot_test_data_full",help="criteo test datasets")
parser.add_argument("--sample_num", "--sample_num", type=int, default=1840617, help="sample number")
parser.add_argument("--result_file","--result_file", type=str,default="../data/output/infer_result.txt", help="result_file")

def create_metrics():
    metrics_list_name = ["auc"]
    auc_metric = Auc("ROC")
    metrics_list = [auc_metric]
    return metrics_list, metrics_list_name

def infer_forward(model, batch, metrics_list):
     batch = [tensor.numpy() for tensor in batch]
     pred = model(batch[1:])
     predict_2d = np.concatenate([1 - pred[0], pred[0]], axis=1)
     metrics_list[0].update(preds=predict_2d, labels=batch[0]) 
     return metrics_list, None


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print(args.magicmind_model + " does not exist.")
        exit()
    
    # model 定义
    model = MMRunner(mm_file=args.magicmind_model, device_id=args.device_id)
    record = Record(args.result_file)
    batch_size = args.batch_size
    dataset_dir = args.dataset_dir
    sample_num = args.sample_num
    print_interval = 10
    file_list = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    dataset = RecDataset(file_list)
    place = paddle.set_device('cpu')
    test_dataloader= DataLoader(
                        dataset,
                        batch_size=batch_size,
                        places=place,
                        drop_last=True,
                        num_workers=16)
    
    metric_list, metric_list_name = create_metrics()

    infer_reader_cost = 0.0
    infer_run_cost = 0.0
    start = time.time()
    interval_begin = time.time()
    reader_start = time.time()
    #we will drop the last incomplete batch when dataset size is not divisible by the batch size
    assert any(test_dataloader(
    )), "test_dataloader is null, please ensure batch size < dataset size!"

    for batch_id, batch in enumerate(test_dataloader()):
        infer_reader_cost += time.time() - reader_start
        infer_start = time.time()
        batch_size = len(batch[0])
        metric_list, tensor_print_dict = infer_forward(model, batch, metric_list) 
        infer_run_cost += time.time() - infer_start

        if batch_id % print_interval == 0:
            metric_str = ""
            for metric_id in range(len(metric_list_name)):
                metric_str += (
                    metric_list_name[metric_id] +
                    ": {:.6f},".format(metric_list[metric_id].accumulate())
                )
            log.info(
                    "batch_id: {}, ".format(
                     batch_id) + metric_str + 
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.2f} ins/s".
                    format(infer_reader_cost / print_interval, 
                           (infer_reader_cost + infer_run_cost) / print_interval,
                           batch_size, 
                           print_interval * batch_size / (time.time() + 0.0001 - interval_begin)))
            record.write("auc result:", False)
            record.write(metric_str,False)
            interval_begin = time.time()
            infer_reader_cost = 0.0
            infer_run_cost = 0.0
        reader_start = time.time()
        if batch_id * batch_size > sample_num:
            break
    end = time.time()
    log.info('total cost time: {}'.format(end-start))