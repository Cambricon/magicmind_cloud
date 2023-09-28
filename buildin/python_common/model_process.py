import argparse
import os
from typing import List, Tuple

from magicmind.python.cc._pywrap_runtime import ITensor
from magicmind.python.cc._pywrap_runtime.enums import DataType
from magicmind.python.runtime import (
    Network,
    Dims,
    BuilderConfig,
    Calibrator,
    RemoteConfig,
    QuantizationAlgorithm,
    Builder,
)
from magicmind.python.runtime.model import Model

from logger import Logger

log = Logger()


MODEL_CONFIG_PARAMS = [
    "input_dims",
    "batch_size",
]

# cambricon-note: add dynamic_shape
MODEL_BUILDER_PARAMS = [
    "mlu_arch",
    "precision",
    "activation_quant_algo",
    "weight_quant_granularity",
    "cluster_num",
    "means",
    "vars",
    "toolchain_path",
    "input_layout",
    "output_layout",
    "dim_range_min",
    "dim_range_max",
    "input_dims",
    "dynamic_shape",
    "computation_preference",
    "type64to32_conversion",
    "conv_scale_fold",
    "build_config",
]

MODEL_CALIBRATION_PARAM = [
    "random_calib_range",
    "device_id",
    "file_list",
    "calibration_data_path",
    "rpc_server",
    "calibration_algo",
    "input_layout",
    "rgb2bgr",
]

MODEL_BUILD_AND_SERIALIZE_PARAM = ["input_dtypes", "output_dtypes", "magicmind_model"]


def extract_params(group: str, args: argparse.Namespace) -> dict:
    """Extract spcified parameters from command line parameters.

    Args:
        group(str): One of {"MODEL_CONFIG_PARAMS", "MODEL_BUILDER_PARAMS",
                            "MODEL_CALIBRATION_PARAM", "MODEL_BUILD_AND_SERIALIZE_PARAM"}
        args(argparse.Namespace): In argument, all parameters from command line.

    Returns:
        - Returns filtered arguments.
    """
    assert isinstance(args, argparse.Namespace), "Got a args with invalid type."
    selected_group = None
    global MODEL_CONFIG_PARAMS, MODEL_BUILDER_PARAMS, MODEL_CALIBRATION_PARAM, MODEL_BUILD_AND_SERIALIZE_PARAM
    assert len(group) > 6, "invalid group name"
    if group.upper() in "MODEL_CONFIG_PARAMS":
        selected_group = MODEL_CONFIG_PARAMS
    elif group.upper() in "MODEL_BUILDER_PARAMS":
        selected_group = MODEL_BUILDER_PARAMS
    elif group.upper() in "MODEL_CALIBRATION_PARAM":
        selected_group = MODEL_CALIBRATION_PARAM
    elif group.upper() in "MODEL_BUILD_AND_SERIALIZE_PARAM":
        selected_group = MODEL_BUILD_AND_SERIALIZE_PARAM
    else:
        raise ValueError("unkonwn group name")
    filtered_args = {
        key: getattr(args, key) for key in selected_group if hasattr(args, key)
    }
    return filtered_args


def check_model_config_params_validness(args: dict):
    """Naive check for the validty of each model configuration parameter.

    Args:
        args(dict): A dict with all parameters for model configuration.
    """
    for item in MODEL_CONFIG_PARAMS:
        if item in args.keys():
            if isinstance(args[item], list):
                assert len(args[item]) > 0, "invalid paramter presented in {}".format(
                    item
                )
        else:
            log.info(
                "parameter {} is not specified, maybe use default value.".format(item)
            )


def param_input_dims_to_mm_dims(input_dims: List[List[int]]) -> List[Dims]:
    """Covert input_dims to magicmind Dims format.
    Args:
        input_dims(List[List[int]]]: In parameter, shape of each input.

    Returns:
        -Returns a list with shape of magicmind Dims for each network input.
    """
    assert (
        isinstance(input_dims, list) and len(input_dims) > 0
    ), "invalid parameter input_dims: " + str(input_dims)
    for dim in input_dims:
        assert isinstance(dim, list) and len(dim) > 0, "invalid parameter"
    return [Dims(e) for e in input_dims]


def print_parameters(args: dict, group: list, description: str):
    """Print a group of arguments.

    Args:
        args(dict): A dict includes all parameters.
        group(list): A list of items to be printed.
        description(str): A description for the group of printed parameters.
    """
    log.info("=" * 30)
    log.info(description)
    for item in group:
        if item in args.keys():
            log.info(item + ": " + str(args[item]))
    log.info("=" * 30)


def config_network(net: Network, args: dict) -> bool:
    """Configure input shape, batch size etc. of the network.

    Args:
        net(Network): Network to be configured.

    Returns:
        - Returns Ture if configure successfully.
    """
    check_model_config_params_validness(args)
    print_parameters(
        args, MODEL_CONFIG_PARAMS, "parameters about network configuration:"
    )
    input_count = net.get_input_count()
    # set input dimension to be fixed args["input_dims"] if dim_range is not specified
    if "input_dims" in args.keys() and args["input_dims"] is not None:
        log.info("config_network: going to reset input dims.")
        mm_input_dims = param_input_dims_to_mm_dims(args["input_dims"])
        assert input_count == len(
            mm_input_dims
        ), "config_network: network has {} inputs, but got {} input_dims from parameter.".format(
            input_count, len(mm_input_dims)
        )
        for i in range(input_count):
            tensor = net.get_input(i)
            assert (
                tensor.set_dimension(mm_input_dims[i]).ok() == True
            ), "Failed to set input dimension"
        log.info("config_network: set input dimensions sucessfully.")

    # set batch size
    if "batch_size" in args.keys() and args["batch_size"] is not None:
        batch_size_list = args["batch_size"]
        assert (
            isinstance(batch_size_list, list) and len(batch_size_list) > 0
        ), "invalid paramter batch_size : " + str(batch_size_list)
        assert (
            len(batch_size_list) == input_count
        ), "config_network: network has {} inputs, but got {} batch sizes from parameter.".format(
            input_count, len(batch_size_list)
        )
        log.info("config_network: going to reset input batch size.")
        for i in range(input_count):
            tensor = net.get_input(i)
            dim = tensor.get_dimension().GetDims()
            if len(dim) > 0:
                dim[0] = batch_size_list[i]
                assert tensor.set_dimension(Dims(dim)).ok(), "Failed to set batch size"
            else:
                log.info(
                    "config_network: Input {} is with rank {}, batch size {} will not be applied.".format(
                        i, len(dim), batch_size_list[i]
                    )
                )
        log.info("config_network: set input batch size sucessfully.")

    log.info("config_network: network was configured successfully.")
    return True


def get_mlu_arches_from_param(args: dict) -> list:
    """Get the mlu architectures specified from command line.

    Args:
        args(dict): All arguments.

    Returns:
        - Returns specified mlu architecture or the defalut value (ARCH_CANDIDATES) if
          no one was specified.
    """
    ARCH_CANDIDATES = ["mtp_372", "tp_322", "mtp_592"]
    assert "mlu_arch" in args.keys(), "mlu_arch is required, but didn't specified."
    mlu_arches = None
    if args["mlu_arch"] is None:
        log.info(
            "mlu archtecture has been set to the default value {} since it was not specified.".format(
                ARCH_CANDIDATES
            )
        )
        mlu_arches = ARCH_CANDIDATES
    elif isinstance(args["mlu_arch"], str):
        assert args["mlu_arch"] in ARCH_CANDIDATES, "invalid mlu architecture"
        mlu_arches = [args["mlu_arch"]]
    elif isinstance(args["mlu_arch"], list):
        assert all(
            [e in ARCH_CANDIDATES for e in args["mlu_arch"]]
        ), "invalid mlu architecture"
        mlu_arches = args["mlu_arch"]
    else:
        raise ValueError("invalid parameter")
    return mlu_arches


def get_precision_json_str_from_param(args: dict) -> str:
    """Convert parameter precision to the json str version.

    Args:
        args(dict): All arguments.

    Returns:
        - Returns a json string representing precision.
    """
    PRECISION_CANDIDATES = [
        "force_float32",
        "force_float16",
        "qint16_mixed_float16",
        "qint8_mixed_float16",
        "qint16_mixed_float32",
        "qint8_mixed_float32",
    ]
    assert "precision" in args.keys(), "precision is required, but didn't specified."
    assert args["precision"] in PRECISION_CANDIDATES, "invalid precision"
    json_str = '{"precision_config":{"precision_mode":"' + args["precision"] + '"}}'
    return json_str


def get_default_cluster_num_of_arches(arch_list: List[str]) -> list:
    """get default cluster number for each architecture.

    Args:
        arch_list(dict): Specified architectures list..

    Returns:
        - Returns a list with cluster number for each architecture.
    """
    # note: default cluster num is not the optimal value, is just a valid one.
    DEFAULT_CLUSTER_NUM_4_EACH_ARCH = {
        "mtp_220": [1],
        "mtp_270": [1],
        "mtp_290": [1],
        "mtp_372": [2,6,8],
        "tp_322": [1],
        "mtp_592": [1],
    }
    default_val = [DEFAULT_CLUSTER_NUM_4_EACH_ARCH[e] for e in arch_list]
    log.info(
        "cluster num for arches {} has been set to default value {} repectively "
        "since it was not specified.".format(
            ",".join(arch_list), ",".join([str(e) for e in default_val])
        )
    )
    return default_val


def set_quant_granul_and_algori(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Set weight quantization granularity and quantization algorithm

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """
    # set weight_quant_granularity
    assert (
        "weight_quant_granularity" in args.keys()
        and args["weight_quant_granularity"] is not None
    ), "weight_quant_granularity is not specified."
    weight_quant_granularity = args["weight_quant_granularity"]
    assert weight_quant_granularity in [
        "per_tensor",
        "per_axis",
    ], "invalid weight_quant_granularity: " + str(weight_quant_granularity)
    weight_quant_granul_str = (
        '{"precision_config": {"weight_quant_granularity": "'
        + weight_quant_granularity
        + '"}}'
    )
    assert config.parse_from_string(weight_quant_granul_str).ok()
    log.info("get_build_config: set weight_quant_granularity successfully.")

    # set symmeteric quantization or assymmeteric quantization
    assert (
        "activation_quant_algo" in args.keys() and args["activation_quant_algo"] is not None
    ), "invalid quantization algorithm"
    activation_quant_algo = args["activation_quant_algo"]
    assert activation_quant_algo in [
        "symmetric",
        "asymmetric",
    ], "invalid quantization algorithm: " + str(activation_quant_algo)
    symmetric_qunat_str = (
        '{"precision_config": {"activation_quant_algo":"' + activation_quant_algo + '"}}'
    )
    assert config.parse_from_string(symmetric_qunat_str).ok()
    log.info(
        "get_build_config: set {} quantization successfully.".format(activation_quant_algo)
    )
    return config


def set_cluster_num(
    config: BuilderConfig, mlu_arch_list: List[str], args: dict
) -> BuilderConfig:
    """Set cluster_num to the specified MLU architecture.

    Args:
        config(BuilderConfig): Configuration for builder.
        mlu_arch_list(List[str]): Architectures to be specified cluster_num.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """
    # set bitmap of visible cluster for each architecture
    # example: "archs": [{"mtp_372": [cluster_num_1, cluster_num_2, cluster_num_3]}]
    cluster_num_list = args["cluster_num"]
    if cluster_num_list is None:
        cluster_num_list = get_default_cluster_num_of_arches(mlu_arch_list)
    assert len(mlu_arch_list) == len(
        cluster_num_list
    ), "{} mlu architectures are specified".format(
        len(mlu_arch_list)
    ) + ", each of which require a cluster_num, but just got {} cluster_num.".format(
        len(cluster_num_list)
    )
    if len(cluster_num_list) > 0 and len(mlu_arch_list) > 0:
        cluster_str = '{"archs": ['
        for i in range(len(mlu_arch_list)):
            cluster_str += '{"' + mlu_arch_list[i] + '": ['
            cluster_str += str(cluster_num_list[i][0])
            for e in range(1, len(cluster_num_list[i])):
                cluster_str += ", " + str(cluster_num_list[i][e])
            cluster_str += "]"
            cluster_str += "}"
        cluster_str += "]}"
    assert config.parse_from_string(cluster_str).ok()
    log.info("get_build_config: set cluster number successfully." + cluster_str)
    return config


def insert_bn_before_first_node(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Insert a batch norm node before the 1st node.

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """
    # an example of mean_var_str:
    # { "insert_bn_before_firstnode": {
    # "0": {"mean": [1.0, 2.0, 3.0, 4.0], "var": [0.01, 0.01, 0.01, 0.01]},
    # "1": {"mean": [1.0, 2.0, 3.0, 4.0], "var": [10.0, 10.0, 10.0, 10.0]}
    #  } }
    # note: data type of input will be changed to UINT8 automatically whatever which datatype
    # you set by parameter input_dtypes once you insert a batchnorm node befor the 1st node.
    assert (
        "means" in args.keys() and "vars" in args.keys()
    ), "means and vars must be specified together, but just got one."
    checker = (
        lambda values: len(values) > 0
        and all([len(a_set_of_value) > 0 for a_set_of_value in values])
        and all(
            [
                type(e) in [float, int]
                for a_set_of_value in values
                for e in a_set_of_value
            ]
        )
    )
    means, vars_ = args["means"], args["vars"]
    assert checker(means), "invalid means"
    assert checker(vars_), "invalid variances"
    assert len(means) == len(vars_), "the number of vars doesn't match that of means"
    mean_var_str = '{"insert_bn_before_firstnode":{'
    for i, (mean, var) in enumerate(zip(means, vars_)):
        mean_var_str += (
            '"{}":'.format(i)
            + "{"
            + '"mean":{}'.format("[" + ",".join([str(e) for e in mean]) + "]")
            + ',"var":{}'.format("[" + ",".join([str(e) for e in var]) + "]")
            + "}"
        )
        mean_var_str += "," if i < len(means) - 1 else "}"
    mean_var_str += "}"
    assert config.parse_from_string(mean_var_str).ok()
    log.info(
        "get_build_config: inserted batchnorm node before the first node successfully."
    )
    return config


def get_channel_opposite_layout(layout_in: str) -> str:
    """Get an opposite layout assoicated with the input layout.

    Args:
        layout_in(str): Input layout.

    Note:
        - Do not set input layout of network to what it already is because the function
            doesn't check the real layout of network input and cannot create a pair of
            {src: dst} with same layout ( for example nchw- > nchw). By default, we think
            you only does convert layout when it needed, and doesn't convert a layout to
            the same.

    Returns:
        - Returns an opposite layout for input layout, such as nchw for nhwc.
    """
    CORRESPONDING_LAYOUT = {"NCT": "NTC", "NCHW": "NHWC", "NCDHW": "NDHWC"}
    CORRESPONDING_LAYOUT.update({v: k for k, v in CORRESPONDING_LAYOUT.items()})
    assert layout_in in CORRESPONDING_LAYOUT.keys(), "unsupported layout"
    return CORRESPONDING_LAYOUT[layout_in]


def set_input_layout(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Set layout for each input of network.

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """
    # note:
    # 1. do not set a layout with axis more than that of input to the input.
    # for example, setting layout NCHW to an input of shape [batch_size, len] with 2 axis
    # is invalid.
    # 2. do not set input layout to what it already is. for example, do not specify
    # converting input layout to nchw if the layout of input already is nchw.
    input_layouts = args["input_layout"]
    assert type(input_layouts) in [list], "invalid input layout: " + str(input_layouts)
    input_layouts = [input_layouts] if isinstance(input_layouts, str) else input_layouts
    input_layout_str = '{"convert_input_layout":{'
    for i, layout in enumerate(input_layouts):
        input_layout_str += (
            '"{}'.format(i)
            + '":{"src":"'
            + get_channel_opposite_layout(layout)
            + '",'
            + '"dst":"'
            + layout
            + '"}'
        )
        input_layout_str += "," if i < len(input_layouts) - 1 else "}"
    input_layout_str += "}"
    assert config.parse_from_string(input_layout_str).ok()
    log.info("get_build_config: set input layout successfully.")
    return config


def set_output_layout(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Set layout for each output of network.

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """
    # note:
    # 1. do not set a layout with axis num more than output axis to the output.
    # for example, setting layout NCHW to an output of shape [batch_size, len] with 2 axis
    # is invalid.
    # 2. do not set output layout to what it already is.
    output_layouts = args["output_layout"]
    assert type(output_layouts) in [list], "invalid input layout"
    output_layouts = (
        [output_layouts] if isinstance(output_layouts, str) else output_layouts
    )
    output_layout_str = '{"convert_output_layout":{'
    for i, layout in enumerate(output_layouts):
        output_layout_str += (
            '"{}'.format(i)
            + '":{"src":"'
            + get_channel_opposite_layout(layout)
            + '",'
            + '"dst":"'
            + layout
            + '"}'
        )
        output_layout_str += "," if i < len(output_layouts) - 1 else "}"
    output_layout_str += "}"
    assert config.parse_from_string(output_layout_str).ok()
    log.info("get_build_config: set output layout successfully.")
    return config


def set_dynamic_or_static_input_shape(
    config: BuilderConfig, args: dict
) -> BuilderConfig:
    """Set dynamic or static input shape for the network.

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """
    if "dynamic_shape" in args.keys() and args["dynamic_shape"] is not None:
        if args["dynamic_shape"] == "true":
            assert config.parse_from_string('{"graph_shape_mutable":true}').ok()
            # cambricon-note: call set_dim_range only when dynamic_shape is true
            set_dim_range(config, args)
        elif args["dynamic_shape"] == "false":
            assert config.parse_from_string('{"graph_shape_mutable":false}').ok()
            log.info("get_build_config: fixed input shape successfully.")
        else:
            log.info("Error:the val of shape_mutable is invalid.")
    #assert config.parse_from_string('{"debug_config": {"fusion_enable": false}}').ok()
    #assert config.parse_from_string('{"debug_config": {"print_ir":  {"print_level": 1}}}').ok()
    return config


def set_dim_range(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Set dim range for the network.

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """
    if (
        "dim_range_min" in args.keys()
        and args["dim_range_min"] is not None
        or "dim_range_max" in args.keys()
        and args["dim_range_max"] is not None
    ):
        # assert is_fixed_input_shape == False, (
        #    "already specified fixed input shape: "
        #    + str(args["input_dims"])
        #    + ", do not specify dim_range for dynamic shape again."
        # )
        assert (
            "dim_range_min" in args.keys()
            and args["dim_range_min"] is not None
            and "dim_range_max" in args.keys()
            and args["dim_range_max"] is not None
        ), "dim_range_min and dim_range_max must be provided together, but just got one of them."

        dim_range_min_list = args["dim_range_min"]
        dim_range_max_list = args["dim_range_max"]

        assert len(dim_range_min_list) == len(
            dim_range_max_list
        ), "dim_range_min does not match dim_range_max"
        dim_range_str = '{"dim_range": {'
        # an example of dim_range_str:
        # { "dim_range": {
        #       "0": {
        #           "min": [1, 224, 224, 3],
        #           "max": [12, 224, 224, 3]
        #       }
        # } }
        for i, (min_list, max_list) in enumerate(
            zip(dim_range_min_list, dim_range_max_list)
        ):
            dim_range_str += (
                '"{}":'.format(i)
                + '{"min":['
                + ",".join([str(e) for e in min_list])
                + "],"
                + '"max":['
                + ",".join([str(e) for e in max_list])
                + "]}"
            )
            dim_range_str += "," if i < len(dim_range_max_list) - 1 else "}"
        dim_range_str += "}"
        assert config.parse_from_string(dim_range_str).ok()
        log.info(
            "get_build_config: set input shape to be mutable by the specified dim_range successfully."
        )
    return config


def set_computation_preference(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Set computation preference for the network.

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """

    assert (
        "computation_preference" in args.keys()
        and args["computation_preference"] is not None
    ), "computation_preference is not specified."

    computation_preference = args["computation_preference"]
    assert computation_preference in [
        "auto",
        "fast",
        "high_precision",
    ], "invalid computation_preference: " + str(computation_preference)
    computation_preference_str = (
        '{"computation_preference":' + '"' + computation_preference + '"' + "}"
    )
    assert config.parse_from_string(computation_preference_str).ok()
    log.info(
        "get_build_config: set computation_preference {} successfully.".format(
            computation_preference
        )
    )
    return config


def set_type64to32_conversion(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Set type64to32_conversion for the network.

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """

    assert (
        "type64to32_conversion" in args.keys()
        and args["type64to32_conversion"] is not None
    ), "opt.type64to32_conversion is not specified."

    type64to32_conversion = args["type64to32_conversion"]
    assert type64to32_conversion in [
        "true",
        "false",
    ], "invalid type64to32_conversion: " + str(type64to32_conversion)

    # example: {"opt_config":{"type64to32_conversion":true}}
    type64to32_conversion_str = (
        '{"opt_config":{"type64to32_conversion":' + type64to32_conversion + "}}"
    )

    assert config.parse_from_string(type64to32_conversion_str).ok()
    log.info(
        "get_build_config: set opt.type64to32_conversion {} successfully.".format(
            type64to32_conversion
        )
    )
    return config


def set_conv_scale_fold(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Set conv_scale_fold for the network.

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """

    assert (
        "conv_scale_fold" in args.keys() and args["conv_scale_fold"] is not None
    ), "opt.conv_scale_fold is not specified."

    conv_scale_fold = args["conv_scale_fold"]
    assert conv_scale_fold in [
        "true",
        "false",
    ], "invalid conv_scale_fold: " + str(conv_scale_fold)

    # example: {"opt_config":{"conv_scale_fold":true}}
    conv_scale_fold_str = '{"opt_config":{"conv_scale_fold":' + conv_scale_fold + "}}"

    assert config.parse_from_string(conv_scale_fold_str).ok()
    log.info(
        "get_build_config: set opt.conv_scale_fold {} successfully.".format(
            conv_scale_fold
        )
    )
    return config


def set_opt_config(config: BuilderConfig, args: dict) -> BuilderConfig:
    """Set opt_config for the network.
       In detail, set opt.type64to32_conversion, opt.conv_scale_fold

    Args:
        config(BuilderConfig): Configuration for builder.
        args(dict): All arguments.

    Returns:
        - Returns a configured BuilderConfig instance.
    """

    set_type64to32_conversion(config, args)
    set_conv_scale_fold(config, args)
    return config


def get_builder_config(args: dict) -> BuilderConfig:
    """Create a BuilderConfig for subsequent model building.

    Args:
        args(dict): All arguments.

    Returns:
        - Returns a BuilderConfig for subsequent model building.
    """
    print_parameters(
        args, MODEL_BUILDER_PARAMS, "parameters about model builder configuration:"
    )
    config = BuilderConfig()
    # set mlu architecture
    mlu_arch_list = get_mlu_arches_from_param(args)
    config.mlu_archs = mlu_arch_list
    log.info("get_build_config: set mlu architecture successfully.")

    # set precision
    assert config.parse_from_string(get_precision_json_str_from_param(args)).ok()
    log.info("get_build_config: set precision successfully.")

    if "int" in args["precision"]:
        config = set_quant_granul_and_algori(config, args)

    # set cluster num
    # the validity of cluster is not checked, please make sure the validity of which before use.
    if "cluster_num" in args.keys():
        config = set_cluster_num(config, mlu_arch_list, args)

    exist_and_not_none_checker = lambda x: x in args.keys() and args[x] is not None

    # set means and vars
    if exist_and_not_none_checker("means") or exist_and_not_none_checker("vars"):
        config = insert_bn_before_first_node(config, args)

    # set cross compiler toolchain
    if exist_and_not_none_checker("toolchain_path"):
        toolchain_path = args["toolchain_path"]
        assert toolchain_path is not None and os.path.exists(
            toolchain_path
        ), "invalid toolchain path"
        assert config.parse_from_string(
            '{"cross_compile_toolchain_path":"' + toolchain_path + '"}'
        ).ok()
        log.info("get_build_config: set cross compile toolchain path successfully.")

    # set layout for input
    if exist_and_not_none_checker("input_layout"):
        config = set_input_layout(config, args)

    # set layout for output
    if exist_and_not_none_checker("output_layout"):
        config = set_output_layout(config, args)

    # set dynamic or static input shape
    set_dynamic_or_static_input_shape(config, args)

    # cambricon-note: add set computation preference
    set_computation_preference(config, args)

    # cambricon-note: add set opt_config
    set_opt_config(config, args)

    # configure the net with the given json
    # note: for a item, the set presented in the json will override the corresponding cofiguration above
    if exist_and_not_none_checker("build_config"):
        json_path = args["build_config"]
        assert isinstance(json_path, str) and os.path.exists(
            json_path
        ), "invalid configuration json path"
        assert config.parse_from_file(json_path).ok()
        log.info(
            "get_build_config: set builder_config by json file successfully. NOTE: the items "
            "configured before has been OVERRIDED with the ones presented in the configuration json."
        )
    log.info("get_build_config: created builder configuraton successfully.")
    return config


def get_nchw_of_input(tesnor: ITensor, layout: str) -> List[int]:
    """Get batch size, channel num, height, width  and channel axis of the tensor

    Args:
        tensor(ITensor): Input tensor to get n, c, h, w and channel_axis.
        layout(str): Layout of the input tensor.

    Returns:
        - Returns a list with n, c, h, w and channel_aixs of the input tensor.
    """
    tensor_dim = tesnor.get_dimension()
    assert layout.upper() in ["NCHW", "NHWC"], "unspported layout: " + layout
    batch_size = tensor_dim.GetDimValue(0)
    get_dim = lambda i: tensor_dim.GetDimValue(i)
    if layout == "NCHW":
        return [batch_size, get_dim(1), get_dim(2), get_dim(3), 1]
    else:
        return [batch_size, get_dim(3), get_dim(1), get_dim(2), 3]


def is_rand_calib_or_fake_calib(
    net: Network, config: BuilderConfig, args: dict
) -> Tuple[bool, bool]:
    """Determine random calibration (calibrate wuth random data) or fake calibration (set the range of
    direction to [-1, 1] directly). If the calibration is  neither random calibration nor fake calibration,
    the network will be calibrated by real calibration samples from dataloader.

    Args:
        net(dict): Network to be calibrated.
        config(BuilderConfig): A BuilderConfig for calibration.
        args(dict): All parameters for calibration.

    Returns:
        - For rand_calib,returns True if calibration is random calibration, otherwise, False. For fake_calib,returns True
          for fake_calib if calibration is fake calibration, otherwise, False.
    """
    rand_calib = False
    fake_calib = False
    if "random_calib_range" in args.keys() and args["random_calib_range"] is not None:
        random_calib_range = args["random_calib_range"]
        assert (
            isinstance(random_calib_range, list) and len(random_calib_range) == 2
        ), "invalid random calibration range, expects a list with [min, max] as range calibration data"
        log.info(
            "calibration: going to calibrate network with custom range [{}]".format(
                str(random_calib_range)
            )
        )
        rand_calib = True
    elif args["file_list"] is None and args["calibration_data_path"] is None:
        log.info(
            "calibration: fill fixed quantization parameters with fake calibration"
        )
        fake_calib = True
        assert config.parse_from_string(
            '{"custom_ranges": {"": {"max": [1], "min": [-1]}}}'
        ).ok()
        log.info("calibration: set fake calibration successfully.")
    else:
        assert (
            len(args["file_list"]) > 0 and args["calibration_data_path"] is not None
        ), "Calibration list file and calibration data path must be provided together."

    input_num = net.get_input_count()
    if "file_list" in args.keys() and args["file_list"] is not None:
        img_list_file_paths = args["file_list"]
        for p in img_list_file_paths:
            assert os.path.exists(p), "file is not existing: {}".format(p)
        assert (
            "calibration_data_path" in args.keys()
            and args["calibration_data_path"] is not None
        ), "parameter calibration_data_path is not provided."
        calibration_data_path = args["calibration_data_path"]
        assert os.path.exists(
            calibration_data_path
        ), "calibration data folder {} is not existing: ".format(calibration_data_path)
        if not rand_calib and not fake_calib and len(img_list_file_paths) != input_num:
            raise ValueError(
                "calibration data should be provided for each input if calibrate"
                + " quantization with real data. expects {} img_list_file for {} inputs,".format(
                    input_num, input_num
                )
                + " but just got {}.".format(len(img_list_file_paths))
            )
    return rand_calib, fake_calib


def check_validity_of_ip(ip: str) -> bool:
    """Naive check for validity of ip address

    Args:
        ip(str): Ip address.

    Returns:
        - Returns Ture if ip address is validity, otherwise False.
    """
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    else:
        return all([int(e) <= 255 and int(e) >= 0 for e in parts])


def build_and_serialize(net: Network, config: BuilderConfig, args: dict) -> bool:
    """Build and serialize the magicmind model.

    Args:
        net(dict): Network to be built.
        config(BuilderConfig): A BuilderConfig for model building.
        args(dict): All parameters for model building and serialization.

    Returns:
        - Returns True if build and serialize successfully, otherwise, False.
    """
    print_parameters(
        args,
        MODEL_BUILD_AND_SERIALIZE_PARAM,
        "parameters about model building and serialization configuration:",
    )
    builder = Builder()
    PARAM_DATA_TYPE_2_MM = {
        "INT8": DataType.INT8,
        "INT16": DataType.INT16,
        "INT32": DataType.INT32,
        "INT64": DataType.INT64,
        "UINT8": DataType.UINT8,
        "UINT16": DataType.UINT16,
        "UINT32": DataType.UINT32,
        "HALF": DataType.FLOAT16,
        "FLOAT": DataType.FLOAT32,
        "BOOL": DataType.BOOL,
        "QINT8": DataType.QINT8,
        "QINT16": DataType.QINT16,
    }

    if "input_dtypes" in args.keys() and args["input_dtypes"] is not None:
        data_types = args["input_dtypes"]
        for data_type in data_types:
            assert data_type in PARAM_DATA_TYPE_2_MM.keys(), (
                "unsupported data type: " + data_type
            )
        input_num = net.get_input_count()
        assert input_num == len(
            data_types
        ), "network has {} inputs, requires {} ".format(
            input_num, input_num
        ) + "data types but just specified {}.".format(
            len(data_types)
        )
        for i in range(input_num):
            tensor = net.get_input(i)
            assert tensor.set_data_type(PARAM_DATA_TYPE_2_MM[data_types[i]]).ok()
            log.info(
                "build_and_serialize: reset input {}'s data type to: ".format(i)
                + args["input_dtypes"][i]
            )
        log.info("build_and_serialize: set input data types successfully.")

    if "output_dtypes" in args.keys() and args["output_dtypes"] is not None:
        data_types = args["output_dtypes"]
        for data_type in data_types:
            assert data_type in PARAM_DATA_TYPE_2_MM.keys(), (
                "unsupported data type: " + data_type
            )
        assert output_num == len(
            data_types
        ), "network has {} outputs, requires {} ".format(
            output_num, output_num
        ) + "data types but just specified {}.".format(
            len(data_types)
        )
        for i in range(output_num):
            tensor = net.get_output(i)
            assert tensor.set_data_type(PARAM_DATA_TYPE_2_MM[data_types[i]]).ok()
            log.info(
                "build_and_serialize: reset output {}'s data type to: ".format(i)
                + args["output_dtypes"][i]
            )
        log.info("build_and_serialize: set output data types successfully.")

    assert (
        "magicmind_model" in args.keys() and args["magicmind_model"]
    ), "path to save magicmind model is not specified."
    magicmind_model = args["magicmind_model"]
    model = builder.build_model("magicmind_model_name", net, config)
    assert isinstance(model, Model)
    log.info("build_and_serialize: built model successfully.")
    assert model.serialize_to_file(magicmind_model).ok()
    log.info("build_and_serialize: serialized model successfully.")

    input_data_types = model.get_input_data_types()
    output_data_types = model.get_output_data_types()
    input_names = model.get_input_names()
    output_names = model.get_output_names()
    input_dimensions = model.get_input_dimensions()
    output_dimensions = model.get_output_dimensions()

    MM_DATA_TYPE_2_STR = {value: key for key, value in PARAM_DATA_TYPE_2_MM.items()}
    log.info("=" * 30)
    log.info("build_and_serialize: model info:")
    for i, (name, type_, shape) in enumerate(
        zip(input_names, input_data_types, input_dimensions)
    ):
        log.info("model input[{}]'s name is {}".format(i, name))
        log.info("model input[{}]'s shape is {}".format(i, str(shape.GetDims())))
        log.info(
            "model input[{}]'s data type is {}".format(i, MM_DATA_TYPE_2_STR[type_])
        )

    for i, (name, type_, shape) in enumerate(
        zip(output_names, output_data_types, output_dimensions)
    ):
        log.info("model output[{}]'s name is {}".format(i, name))
        log.info("model output[{}]'s shape is {}".format(i, str(shape.GetDims())))
        log.info(
            "model output[{}]'s data type is {}".format(i, MM_DATA_TYPE_2_STR[type_])
        )
    log.info("=" * 30)
