# -*- coding: utf-8 -*-
from magicmind.python.runtime.parser import Parser
import magicmind.python.runtime as mm
import argparse

def do_calibrate(network, calib_data, config, precision):
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None

    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    assert config.parse_from_string(
        """{"precision_config": {"precision_mode": "%s"}}"""%(precision)).ok()
    assert calibrator.calibrate(network, config).ok()

def construct_model(args, reload_by_serilize=True):
    # init builder, network, builder_config and parser
    builder = mm.Builder()
    network = mm.Network()
    config = mm.BuilderConfig()
    parser = Parser(mm.ModelKind.kTensorflow)

    # get input dims from network
    parser.set_model_param("tf-model-type", "tf-graphdef-file")
    parser.set_model_param("tf-graphdef-inputs", ["input_1"])
    parser.set_model_param("tf-graphdef-outputs", ["pred_pose/mul_24"])
    
    assert parser.parse(network, args.tf_pb).ok()
    assert network.get_input(0).set_dimension(mm.Dims((args.batch_size, args.img_size[0], args.img_size[1], 3))).ok()
    config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}')
    config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}')
    config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}')
    if args.shape_mutable=='true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()  

    model = builder.build_model("mm_model", network, config)
    assert model.serialize_to_file(args.mm_model).ok()
    print(args.mm_model," was saved successfully in the current path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_pb', type=str, default='../data/models/psenet.pb', help='tf_pb path(s)')
    parser.add_argument('--mm_model', type=str, default='../data/', help='saved .mm model name')
    parser.add_argument('--precision', type=str, default='force_float32', help='precision')
    parser.add_argument('--shape_mutable', type=str, default="false", required=True ,help='shape_mutable')
    parser.add_argument('--img_size', type=list, default=[64, 64], help='inference size (pixels)')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch, default:1') 
    args = parser.parse_args()    

    if args.precision != "force_float32":
        print("FSA-Net Only Support FP32(force_float32) Mode !")
        exit(0)
    construct_model(args)

