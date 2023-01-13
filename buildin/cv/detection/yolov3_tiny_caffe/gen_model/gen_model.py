import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import numpy as np
import glob
import os 
import argparse
import cv2
from calibrator import CalibData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "c3d caffe model calibrartion and build")
    parser.add_argument('--precision', type=str,   default='force_float16', required=True ,help='Quant_mode')
    parser.add_argument('--batch_size', type=int,   default=8, required=True ,help='batch_size')
    parser.add_argument('--shape_mutable', type=str, default="false", required=True ,help='shape_mutable')
    parser.add_argument('--datasets_dir', type=str, default="false", required=True ,help='datasets_dir')
    parser.add_argument('--caffe_prototxt', type=str, default="false", required=True ,help='caffe_prototxt')
    parser.add_argument('--caffe_model', type=str, default="false", required=True ,help='caffe_model')
    parser.add_argument('--mm_model', type=str, default="", required=True ,help='caffe_model')
    args = parser.parse_args()

    DEV_ID = 0
    PAD_VALUE = 128
    BATCH_SIZE = args.batch_size
    INPUT_SIZE = (416, 416) # h x w
    IMAGE_DIR = args.datasets_dir #'val2017'
    PROTOTXT = args.caffe_prototxt
    CAFFEMODEL = args.caffe_model
    ANCHORS = [81,82,  135,169,  344,319, 23,27,  37,58, 81,82]
    MM_MODEL = args.mm_model
    CALIB_SAMPLES_DIR = IMAGE_DIR
    MAX_CALIB_SAMPLES = BATCH_SIZE

    parser = Parser(mm.ModelKind.kCaffe)
    network = mm.Network()
    assert parser.parse(network, CAFFEMODEL, PROTOTXT).ok()
    assert network.get_input(0).set_dimension(mm.Dims((BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]))).ok()

    config = mm.BuilderConfig()
    precision_json_str = '{"precision_config" : { "precision_mode" : "%s" }}'%args.precision
    assert config.parse_from_string(precision_json_str).ok()

    assert config.parse_from_string("{\"opt_config\":{\"type64to32_conversion\":true}}").ok()
    assert config.parse_from_string("{\"opt_config\":{\"conv_scale_fold\":true}}").ok()
    if args.shape_mutable=='true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 416, 416], "max": [32, 3, 416, 416]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    # 指定设备
    assert config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}').ok()
    assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    assert config.parse_from_string('{"insert_bn_before_firstnode": {"0": {"mean": [0, 0, 0], "var": [65025, 65025, 65025]}}}').ok()
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()

    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        if DEV_ID >= dev_count:
            print("Invalid DEV_ID set!")
        dev = mm.Device()
        dev.id = DEV_ID
        assert dev.active().ok()

        if args.precision != "force_float16" and args.precision != "force_float32":
            print("Calibraing...")
            calib_data = CalibData(mm.Dims((BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1])), MAX_CALIB_SAMPLES, CALIB_SAMPLES_DIR,PAD_VALUE)
            calibrator = mm.Calibrator([calib_data])
            assert calibrator is not None
            assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
            assert calibrator.calibrate(network, config).ok()
            print("Calibra Done!")

        perms = [0, 2, 3, 1]  # 0 : N, 1 : C, 2 : H, 3 : W
        const_node = network.add_i_const_node(mm.DataType.INT32, mm.Dims([len(perms)]), np.array(perms, dtype=np.int32))
        output_tensors = []
        for i in range(network.get_output_count()):
            tensor = network.get_output(i)
            permute_node = network.add_i_permute_node(tensor, const_node.get_output(0))
            output_tensors.append(permute_node.get_output(0))
        output_count = network.get_output_count()
        for i in range(output_count):
            network.unmark_output(network.get_output(0))

        # anchors
        bias_buffer = ANCHORS
        bias_node = network.add_i_const_node(mm.DataType.FLOAT32, mm.Dims([len(bias_buffer)]),
            np.array(bias_buffer, dtype=np.float32))
        detect_out = network.add_i_detection_output_node(
            output_tensors, bias_node.get_output(0))
        #添加后处理算子
        detect_out.set_layout(mm.Layout.NONE, mm.Layout.NONE)
        detect_out.set_algo(mm.IDetectionOutputAlgo.YOLOV3)
        detect_out.set_confidence_thresh(0.001)
        detect_out.set_nms_thresh(0.45)
        detect_out.set_scale(1.0)
        detect_out.set_num_coord(4)
        detect_out.set_num_class(80)
        detect_out.set_num_entry(5)
        detect_out.set_num_anchor(3)
        detect_out.set_num_box_limit(256)
        detect_out.set_image_shape(INPUT_SIZE[0], INPUT_SIZE[1])
        detection_output_count = detect_out.get_output_count()
        for i in range(detection_output_count):
            network.mark_output(detect_out.get_output(i))

        builder = mm.Builder()
        assert builder is not None
        mm_model = builder.build_model("magicmind model", network, config)
        assert mm_model is not None
        assert mm_model.serialize_to_file(MM_MODEL).ok()
