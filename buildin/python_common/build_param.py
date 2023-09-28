import argparse
import distutils.util


def get_argparser():
    """Parse commond line parameters.

    Returns:
        - Returns a argparse.Namespace with all command line parameters.
    """
    parent_parser = argparse.ArgumentParser(add_help=False)
    # parameters about network configuration
    net_config = parent_parser.add_argument_group(
        title="net_config", description="parameters for network configuration"
    )
    net_config.add_argument(
        "--input_dims",
        dest="input_dims",
        type=int,
        nargs="+",
        action="append",
        help="Input shapes by order. -1 represents dynamic axis. Specify dims for "
        + "for each input by --input_dims 1 3 224 224 --input_dims 1 3 24 24 --input_dims "
        + "... if there are multiple inputs.",
    )
    net_config.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        nargs="+",
        help="Input batchsize by order, will override all highest"
        + "dimensions for all inputs and not affect unrank/scalar inputs.",
    )
    net_config.add_argument(
        "--rgb2bgr",
        dest="rgb2bgr",
        default=False,
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="convert RGB to BGR for first layer's Conv/BatchNorm/Scale of network",
    )

    # parameters about model builder
    model_builder = parent_parser.add_argument_group(
        title="model_builder", description="parameters for model building"
    )
    model_builder.add_argument(
        "--mlu_arch",
        dest="mlu_arch",
        type=str,
        nargs="+",
        default="mtp_372",
        choices=["mtp_220", "mtp_270", "mtp_290", "mtp_372", "tp_322", "mtp_592"],
        help="Target arch for mlu dev. Unset means all.",
    )
    model_builder.add_argument(
        "--cluster_num",
        dest="cluster_num",
        type=int,
        nargs="+",
        action="append",
        help="Allow users to flexibly specify the cluster.",
    )
    model_builder.add_argument(
        "--precision",
        dest="precision",
        type=str,
        default="force_float32",
        choices=[
            "force_float32",
            "force_float16",
            "qint16_mixed_float16",
            "qint8_mixed_float16",
            "qint16_mixed_float32",
            "qint8_mixed_float32",
        ],
        help="Mix precision mode.",
    )
    model_builder.add_argument(
        "--means",
        dest="means",
        type=float,
        nargs="+",
        action="append",
        help="Means for inputs by order. MUST input with vars.",
    )
    model_builder.add_argument(
        "--vars",
        dest="vars",
        type=float,
        nargs="+",
        action="append",
        help="Vars for inputs by order. MUST input with means.",
    )
    model_builder.add_argument(
        "--toolchain_path",
        dest="toolchain_path",
        type=str,
        default=None,
        help="Cross compile toolchain path for tp_322 and mtp_220.",
    )
    model_builder.add_argument(
        "--input_layout",
        dest="input_layout",
        type=str,
        nargs="+",
        choices=["NCHW", "NHWC", "NCT", "NTC", "NCDHW", "NDHWC"],
        help="Convert input layouts from channel last to channel second"
        + "(or the opposite) by order.",
    )
    model_builder.add_argument(
        "--output_layout",
        dest="output_layout",
        type=str,
        nargs="+",
        choices=["NCHW", "NHWC", "NCT", "NTC", "NCDHW", "NDHWC"],
        help="Output data types by order.",
    )
    model_builder.add_argument(
        "--dim_range_min",
        dest="dim_range_min",
        type=int,
        nargs="+",
        action="append",
        help="lower dimension boundary for each input. specify like "
        "--dim_range_min input0_shape[0] input0_shape[1] input0_shape[2] input0_shape[3]"
        "--dim_range_min input1_shape[0] input1_shape[1] input1_shape[2] input1_shape[4]"
        "... --dim_range_min inputn_shape[0] inputn_shape[2]...",
    )
    model_builder.add_argument(
        "--dim_range_max",
        dest="dim_range_max",
        type=int,
        nargs="+",
        action="append",
        help="upper dimension boundary for each input. specify like "
        "--dim_range_max input0_shape[0] input0_shape[1] input0_shape[2] input0_shape[3]"
        "--dim_range_max input1_shape[0] input1_shape[1] input1_shape[2] input1_shape[4]"
        "... --dim_range_max inputn_shape[0] inputn_shape[2]...",
    )
    model_builder.add_argument(
        "--build_config",
        dest="build_config",
        type=str,
        default=None,
        help="Additional json build config for build. The param in config json will"
        + "override the same setted arg params.",
    )

    # cambricon-note: add dynamic_shape
    model_builder.add_argument(
        "--dynamic_shape",
        dest="dynamic_shape",
        type=str,
        default="true",
        choices=["true", "false"],
        help="used to control graph_shape_mutable is true or false",
    )
    model_builder.add_argument(
        "--computation_preference",
        dest="computation_preference",
        type=str,
        default="auto",
        choices=["auto", "fast", "high_precision"],
        help="Specifies the computation mode of operations in network.",
    )
    model_builder.add_argument(
        "--type64to32_conversion",
        dest="type64to32_conversion",
        type=str,
        default="false",
        choices=["false", "true"],
        help="Specifies whether to enable the data conversion from 64-bit to 32-bit. This option should be enabled when parsing a framework model containing bit data (not supported by MagicMind) with IParser. This configuration is also valid for the quantized calibration. If there is a risk of accuracy overflow during this conversion, MagicMind will emit warning messages.",
    )
    model_builder.add_argument(
        "--conv_scale_fold",
        dest="conv_scale_fold",
        type=str,
        default="false",
        choices=["false", "true"],
        help="Whether to prepend the multiplication and addition operations to the convolution weights after the convolution operation. This configuration is also valid for the quantization calibration. If you want to use this optimization for quantization, you need to set this option in in advance during the quantization calibration phase to ensure that the quantization parameters during the calibration process are consistent with the actual data distribution. Note that this optimization may have an impact on the range of the weight distribution and turning on this optimization in lower bit-width uniform quantization algorithms may cause the accuracy not to meet the requirements.",
    )

    # parameters about model calibration
    calibration = parent_parser.add_argument_group(
        title="model_calibration", description="parameters for model calibration"
    )
    calibration.add_argument(
        "--device_id",
        dest="device_id",
        type=int,
        default=0,
        help="MLU device id, used for calibration.",
    )
    calibration.add_argument(
        "--random_calib_range",
        dest="random_calib_range",
        type=float,
        nargs="*",
        help="Set random range for calibration. Will override path and filelist.",
    )
    calibration.add_argument(
        "--file_list",
        dest="file_list",
        type=str,
        nargs="+",
        help="Input file list path by order. For calibration only. MUST "
        + "input with calibration_data_path.",
    )
    calibration.add_argument(
        "--calibration_data_path",
        dest="calibration_data_path",
        type=str,
        default=None,
        help="Directory for calibration data. MUST input with file_list.",
    )
    calibration.add_argument(
        "--calibration",
        dest="calibration",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default="false",
        help="To do calibration or not. Will use range [-1,1] and skip "
        + "calibration if no file or range is set.",
    )
    calibration.add_argument(
        "--rpc_server",
        dest="rpc_server",
        type=str,
        default=None,
        help="Set remote address for calibration.",
    )
    calibration.add_argument(
        "--calibration_algo",
        dest="calibration_algo",
        type=str,
        default="linear",
        choices=["linear", "eqnm"],
        help="Set quantization algorithm for calibration.",
    )
    calibration.add_argument(
        "--weight_quant_granularity",
        dest="weight_quant_granularity",
        type=str,
        default="per_tensor",
        choices=["per_tensor", "per_axis"],
        help="quantization granularity for weights.",
    )
    calibration.add_argument(
        "--activation_quant_algo",
        dest="activation_quant_algo",
        type=str,
        default="symmetric",
        choices=["symmetric", "asymmetric"],
        help="specify symmetric quantization or asymmetric quantization.",
    )

    # parameters about model building and serialization
    build_and_serilaize = parent_parser.add_argument_group(
        title="model_build_and_serialize",
        description="parameters for model building and serialization",
    )
    build_and_serilaize.add_argument(
        "--input_dtypes",
        dest="input_dtypes",
        type=str,
        choices=[
            "INT8",
            "INT16",
            "INT32",
            "UINT8",
            "UINT16",
            "UINT32",
            "HALF",
            "FLOAT",
            "BOOL",
            "QINT8",
            "QINT16",
        ],
        nargs="+",
        help="Input data types by order for inference (will not affect"
        + "calibration).",
    )
    build_and_serilaize.add_argument(
        "--output_dtypes",
        dest="output_dtypes",
        type=str,
        choices=[
            "INT8",
            "INT16",
            "INT32",
            "UINT8",
            "UINT16",
            "UINT32",
            "HALF",
            "FLOAT",
            "BOOL",
            "QINT8",
            "QINT16",
        ],
        nargs="+",
        help="Output data types by order.",
    )
    build_and_serilaize.add_argument(
        "--magicmind_model",
        dest="magicmind_model",
        type=str,
        default="./model",
        help="File path for output serialization model file.",
    )

    parent_parser.add_argument(
        "--plugin_path",
        dest="plugin_path",
        type=str,
        nargs="*",
        help="Plugin library paths to link with.",
    )

    # parameters about model calibration
    framework_parser = parent_parser.add_argument_group(
        title="framework_parser", description="parameters for pytorch/tf/caffe/onnx parser"
    )

    framework_parser.add_argument(
        "--tf_pb",
        dest="tf_pb",
        type=str,
        default=None,
        help="TensorFlow pb file for TensorFlow parser.",
    )
    framework_parser.add_argument(
        "--input_names",
        dest="input_names",
        type=str,
        nargs="+",
        default=None,
        help="Input names for TensorFlow parser.",
    )
    framework_parser.add_argument(
        "--output_names",
        dest="output_names",
        type=str,
        nargs="+",
        default=None,
        help="Output names for TensorFlow parser.",
    )
    # pytorch
    framework_parser.add_argument(
        "--pytorch_pt",
        dest="pytorch_pt",
        type=str,
        default=None,
        help="PyTorch pt file path for PyTorch parser.",
    )
    framework_parser.add_argument(
        "--pt_input_dtypes",
        dest="pt_input_dtypes",
        type=str,
        nargs="+",
        choices=[
            "INT8",
            "INT16",
            "INT32",
            "UINT8",
            "UINT16",
            "UINT32",
            "HALF",
            "FLOAT",
            "BOOL",
            "QINT8",
            "QINT16",
        ],
        default="FLOAT",
        help="Input data types by order for parsing PyTorch pt",
    )
    # onnx
    framework_parser.add_argument(
        "--onnx",
        dest="onnx",
        type=str,
        default=None,
        help="Onnx file path for ONNX parser.",
    )
    # caffe
    framework_parser.add_argument(
        "--prototxt",
        dest="prototxt",
        type=str,
        default=None,
        help="prototxt file path for Caffe parser.",
    )
    framework_parser.add_argument(
        "--caffemodel",
        dest="caffemodel",
        type=str,
        default=None,
        help="Caffemodel file path for Caffe parser.",
    )
    return parent_parser
