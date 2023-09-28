import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "model build")
    parser.add_argument('--precision', type=str,   default='force_float16', required=True ,help='Quant_mode')
    parser.add_argument('--batch_size', type=int,   default=1, required=True ,help='batch_size')
    parser.add_argument('--shape_mutable', type=str, default="", required=True ,help='shape_mutable')
    parser.add_argument('--onnx_model', type=str, default="", required=True ,help='onnx_model')
    parser.add_argument('--mm_model', type=str, default="", required=True ,help='mm_model')
    parser.add_argument('--input_size', type=int, default=224, required=True ,help='input_size')
    args = parser.parse_args()

    DEV_ID = 0
    BATCH_SIZE = args.batch_size
    ONNX_MODEL = args.onnx_model
    MM_MODEL = args.mm_model
    INPUT_SIZE=(args.input_size,args.input_size) #固定输入尺寸
    
    network = mm.Network()
    parser = Parser(mm.ModelKind.kOnnx)
    assert parser.parse(network, ONNX_MODEL).ok()
    assert network.get_input(0).set_dimension(mm.Dims((BATCH_SIZE, 3, INPUT_SIZE[0], INPUT_SIZE[1]))).ok()
    config = mm.BuilderConfig()
    precision_json_str = '{"precision_config" : { "precision_mode" : "%s" }}'%args.precision
    assert config.parse_from_string(precision_json_str).ok()
    assert config.parse_from_string("{\"opt_config\":{\"type64to32_conversion\":true}}").ok()
    assert config.parse_from_string("{\"opt_config\":{\"conv_scale_fold\":false}}").ok()
    if args.shape_mutable == 'true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{ \
        "dim_range": {  \
        "0": {  \
            "min": [1,3,224,224],  \
            "max": [16,3,1344,1920]  \
        }}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    # 指定设备
    assert config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}').ok()
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()

    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        if DEV_ID >= dev_count:
            print("Invalid DEV_ID set!")
            abort()
        dev = mm.Device()
        dev.id = DEV_ID
        assert dev.active().ok()
        builder = mm.Builder()
        assert builder is not None
        mm_model = builder.build_model("magicmind model", network, config)
        assert mm_model is not None
        assert mm_model.serialize_to_file(MM_MODEL).ok()
