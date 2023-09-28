import magicmind.python.runtime as mm
import os
import time
from criteo_reader import RecDataset
from logger import Logger
from calibrator import CalibData
import paddle
from paddle.io import DataLoader

log = Logger()

from framework_parser import OnnxParser
from build_param import get_argparser
from model_process import (
    extract_params,
    config_network,
    get_builder_config,
    build_and_serialize,
)

def get_network(args):
    parser = OnnxParser(args)
    network = parser.parse()
    return network

def get_args():
    arg_parser = get_argparser()
    return arg_parser.parse_args()

def calibrate(args, network: mm.Network, config: mm.BuilderConfig): 
    sparse_inputs_slots = 27
    dense_input_dim=13
    # 创建量化工具并设置量化统计算法
    sample_data = []
    data_dir = args.calibration_data_path
    batch_size = args.input_dims[0][0]
    MAX_SAMPLES = batch_size
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    dataset = RecDataset(file_list)
    place = paddle.set_device('cpu')
    test_dataloader= DataLoader(
                        dataset,
                        batch_size=batch_size,
                        places=place,
                        drop_last=True,
                        num_workers=0)
    input_data = []
    for batch_id, batch in enumerate(test_dataloader()):
        batch_numpy_list = [tensor.numpy() for tensor in batch]
        input_data = batch_numpy_list[1:]
        if batch_id == 0:
            break
    for i in range(sparse_inputs_slots-1):
        sample_data.append(CalibData(mm.Dims((batch_size, 1)), i, MAX_SAMPLES,input_data))
    sample_data.append(CalibData(mm.Dims((batch_size, dense_input_dim)), sparse_inputs_slots-1, MAX_SAMPLES,input_data))
    calibrator = mm.Calibrator(sample_data)
    #assert calibrator is not None
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    assert calibrator.calibrate(network, config).ok()

def main():
    begin_time = time.time()
    args = get_args()
    network = get_network(args)

    # configure network, such as input_dim, batch_size ...
    config_args = extract_params("MODEL_CONFIG", args)
    config_network(network, config_args)

    # create build configuration
    builder_args = extract_params("MODEL_BUILDER", args)
    build_config = get_builder_config(builder_args)
    assert build_config.parse_from_string('{"debug_config": {"fusion_enable": false}}').ok()

    if args.precision.find("qint") != -1:
        log.info("Do calibrate...")
        calibrate(args, network, build_config)
    log.info("Build model...")
    
    # 生成模型并导出
    build_and_serialize_args = extract_params("MODEL_BUILD_AND_SERIALIZE", args)
    build_and_serialize(network, build_config, build_and_serialize_args)

    end_time = time.time()
    log.info("gen_model time cost:{:.3f}s".format(end_time - begin_time))


if __name__ == "__main__":
    main()









 
