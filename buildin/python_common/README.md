# python Parser&Builder 公共组件

[TOC]
## 文件概述

本文件夹内的文件可用于编译及构建MagicMind网络，仅限于Python端。这些文件可用在magicmind_cloud modelzoo内的各类网络中，使用这些文件内已有的公共函数可减少您的网络移植工作量。具体可参考`cv/detection/yolov5_v6_1_pytorch/gen_model`内的`python`文件。

| 文件名称  | 用途     |
| ------ | -------------------  |
| build_param.py | 包含模型编译及量化、第三方框架文件解析所需的大部分公共参数    | 
| common_calibrate.py | 接受量化数据，网络(network)，MagicMind配置(config)，完成MagicMind量化操作     | 
| framework_parser.py | 将第三方框架的文件解析为MagicMind可用的Network  | 
| logger.py | 日志打印模块    | 
| model_process.py|   包含MagicMind模型编译、配置、序列化所需的公共函数  | 

## 详细说明
### build_param.py 详细说明

该脚本内涵盖了模型编译及量化、第三方框架文件解析所需的大部分公共参数，脚本内首先定义了一组基础的公共argparser，随后以公共argparser为”父类“，定义了4种不同的”子类“，即各个框架的sub_argparser，最后会返回4种sub_argparser（分别为tf_parser, pytorch_parser, caffe_parser, onnx_parser）。根据您正在移植的网络所属原框架类型，从该脚本中获取与您框架相关的sub_argparser（如从pytorch移植yolov5到magicmind，则您只需接收pytorch_parser），随后可在返回的sub_argparser的基础上继续添加自定义参数，如网络特定的一些参数。
#### 1. 通用参数说明

| 参数名称 | 是否必需 | 输入格式 | 参数描述 | 注意事项 |
|---|---|---|---|---|
| input_dims            | 否 | --input_dims n0 c0 h0 w0 --input_dims n1 c1 h1 w1 ...   | 模型输入维度 | 1. 多输入时，以--input_dims n0 c0 h0 w0 --input_dims n1 c1 h1 w1 ... 的形式输入每个输入维度 （一个--input_dims只能跟一个输入的形状）。2. PyTorch与TensorFlow模型中没有明确的输入维度信息，如果不填入此参数，则需要运行期指定。3. 动态输入（即指定dim_range）时无需指定该参数。 |
| batch_size            | 否 | --batch_size batch1 batch2 ...          | 设置模型所有输入的batch数目。| 1. 默认可以不设置。2. 优先级高于输入形状。3. 模型输入有多个时，--batch_size后依次跟每个输入的batch size，各个batch size之间以空格分隔。|
| rgb2bgr               | 否 | --rgb2bgr 0/1/True/true/False/false          | 将Conv卷积网络首层参数从RGB格式转为BGR格式，若首层Conv前有乘加算子，同样会进行转换。不能对非卷积网络使用。 | - |
| mlu_arch              | 否 | --mlu_arch mtp_1 mtp_2 ...         | 指定部署的MLU设备平台 | 1. 默认进行"mtp_372"编译。 2. 指定多个平台时，--mlu_arch后依次跟每个arch的名称，各个名称之间以空格分隔。3. 具体支持配置语义同magicmind.python.runtime.BuilderConfig文档。 |
| cluster_num           | 否 | --cluster_num cluster_num_for_arch1 --cluster_num cluster_num_for_arch2 ...    | 指定模型使用的cluster_num | 默认为1。1. cluster_num是指在模型部署后运行时的硬件资源数量。 2. cluster_num可以配置多个值，如果用户设置的cluster_num的值小于或等于实际获得的cluster_num，它将首先匹配完全相等的值，然后选择最接近的值。如果用户设置的cluster_num的所有值都大于实际获得的cluster_num，将在运行时阶段报告错误。3. 为多个archs设置cluster_num时需要保证设置的cluster_num的数量要与archs数量一致，每个cluster_num均需以 --cluster_num cluster_num_for_arch的形式指定（即一个cluster_num后面只能跟一个arch的cluster_num值）。|
| precision             | 否 | --precision mode                  | 网络精度模式 | 1. 默认使用forced_float32。2. 具体支持配置语义同magicmind.python.runtime.BuilderConfig文档。 |
| means                 | 否 | --means m0_chan0 m0_chan1 m0_chan2 --means m1_chan0 m1_chan1 m1_chan2 ...       | 模型输入各通道的均值， 做减均值操作：(input - mean) | 1. 数据类型为浮点。2. 具体支持配置语义同magicmind.python.runtime.BuilderConfig文档。3. 当输入有多个的时候，每个输入means均以--means m_chan0 m_chan1 m_chan2的形式提供（即一个--means后只能跟一个输入的三个通道的均值）。 |
| vars                 | 否 | --vars var0_chan0 var0_chan1 var0_chan2 --vars var1_chan0 var1_chan1 var1_chan2 ...      | 模型输入各通道的方差， 做除标准差操作：(input - mean) / std，方差val就是std² | 1. 数据类型为浮点。2. 具体支持配置语义同magicmind.python.runtime.BuilderConfig文档。3. 当输入有多个的时候，每个输入vars均以--vars var_chan0 var_chan1 var_chan2的形式提供（即一个--vars后只能跟一个输入的三个通道的方差）。 |
| toolchain_path        | 否 | --toolchain_path /path/to/toochain| 指定交叉编译工具链的路径 | 1. 默认值为None。2. 推荐gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu版本。 |
| input_layout          | 否 | --input_layout layout1 layout2 ...   | MagicMind编译后模型输入布局 | 1. 编译过程中将输入布局由通道后置转为通道前置（或者反过来），给定的参数即为转换后的布局，举例：给如NTC，代表希望将模型输入由NCT转为NTC。2. 在本示例中请勿重复设置输入已经是的数据布局，如模型已经是NTC时，请勿再将该项设置为NTC（该重复设置场景在本示例中的语义为输入当前layout为NCT，将从NCT转为NTC，但当前输入已是NTC而非NCT），否则会出现错误。 |
| output_layout         | 否 | --output_layout layout1 layout2 ...   | MagicMind编译后模型输出布局 | 1. 编译过程中将输出布局由通道后置转为通道前置（或者反过来），给定的参数为转换后的布局，举例：给如NCHW，代表希望将模型输出由NHWC转为NCHW。 2. 和input_layout一样，在本示例中请勿重复设置输入已经是的数据布局。|
| shape_mutable         | 否 | --shape_mutable True  | 用于控制graph_shape_mutable的值。 |若为True，则表示此时网络的输入形状是可变的，否则表示网络输入形状是不可变的。|
| dim_range_min                 | 否 | --dim_range_min in0_n in0_c in0_h in0_w --dim_range_min  in1_n in1_c in1_h in1_w  ...      | 模型为动态输入时，动态输入的最小范围 | 1. 当输入有多个的时候，每个输入的维度下界均以--dim_range_min in_n in_c in_h in_w的形式提供（即一个--dim_range_min后只能跟一组维度）。2. 本示例中设置该项后会将graph_shape_mutable自动为True。3. 该项目需要和 dim_range_min一起指定使用。4. 只允许一个维度变化。|
| dim_range_max                 | 否 | --dim_range_max in0_n in0_c in0_h in0_w --dim_range_max  in1_n in1_c in1_h in1_w  ...      | 模型为动态输入时，动态输入的最大范围 | 1. 当输入有多个的时候，每个输入的维度上界均以--dim_range_max in_n in_c in_h in_w的形式提供（即一个--dim_range_min后只能跟一组维度）。2. 本示例中设置该项后会将graph_shape_mutable自动为True。3. 该项目需要和 dim_range_min一起指定使用。4. 只允许一个维度变化。|
| build_config          | 否 | --build_config path/to/file       | BuildConfig配置json文件 | 1. 以json文件的形式配置编译参数。2. 具体支持配置语义同magicmind.python.runtime.BuilderConfig文档。3. 指定json后，json文件中的配置会覆盖对应的命令行参数配置。 |
| random_calib_range    | 否 | --random_calib_range min max      | 量化校准使用随机数据，决定随机数据的分布上下界。优先级高于校准样本文件。 |
| file_list             | 否 | --file_list path/to/file          | 量化校准文件列表 | 1. 文件包含所有校准样本的文件名列表，用来做量化校准使用。2. 每行的格式为文件名 shape[n, c, h, w]，其中shape为可选项，当有shape项时每一个文件名代表一batch数据，各文件的维度可以不同（对应网络动态形状）。|
| calibration_data_path | 否 | --calibration_data_path path      | 表示量化校准用数据集目录。| - |
| calibration           | 否 | --calibration 0/1/True/true/False/false      | 量化校准开关 | 默认关，如果打开可以配合量化校准数据集路径和文件列表，亦可以不给入任何数据集和路径直接填入默认量化参数（只保证编译，不保证精度）。 |
| rpc_server           | 否 | --rpc_server ip_address     | 带有MLU卡的远端（Remote）设备IP地址 | - |
| calibration_algo      | 否 | --calibration_algo linear/eqnm    | 量化统计算法 | 默认linear，选择量化校准时使用的量化算法。 |
| device_id      | 否 | --device_id 0    | 量化时指定的MLU设备号 | 默认为0，即选择第0号MLU设备 |
| weight_quant_granularity      | 否 | --weight_quant_granularity per_tensor/per_axis    | 量化粒度 | 1. 默认值为per_tensor。2. weight_quant_granularity 对 IConvDepthwiseNode 不适⽤。 |
| activation_quant_algo      | 否 | --activation_quant_algo symmetric/asymmetric    | 量化算法 | 默认值为symmetric。|
| input_dtypes          | 否 | --input_dtypes type1 type2 ...     | 模型输入类型 | 多输入之间以空格作为分隔符。|
| magicmind_model       | 否 | --magicmind_model path/to/file    | 输出离线模型文件 | 默认为./converted.mm_model。 |
| plugin_path                | 否 | --plugin_path /path/to/plugin_1 /path/to/plugin_1 ...        | 指定plugin算子的库地址 | 多plugin库之间以空格作为分隔符。 |

#### 2. 第三方框架模型特有参数说明
#### 2.1 TensorFlow 模型特有参数说明

| 参数名称 | 是否必需 | 输入格式 | 参数描述 | 注意事项 |
|---|---|---|---|---|
| tf_pb                 | 是 | --tf_pb path/to/file              | TensorFlow框架模型文件路径 | 1. pb模型格式原生没有输入输出信息，需要和input_names/output_names一起提供。2. 需将框架名指定为tensorflow。3. MagicMind还支持GraphDef序列化的string模型和saved model模型，具体可参考寒武纪MagicMind用户手册。 |
| input_names           | 是 | --input_names name1  name2 ...      | TensorFlow模型的输入名称 | 1. 需将框架名指定为tensorflow。2. pb模型输入有多个的时候，--input_names后依次跟每个输入的名称，各个名称之间以空格分隔。3. 需要和tf_pb及output_names一起提供。 |
| output_names          | 是 | --output_names name1 name2 ...     | TensorFlow模型的输出名称  |  1. 需将框架名指定为tensorflow。2. pb模型输出有多个的时候，--output_names后依次跟每个输出的名称，各个名称之间以空格分隔。3. 需要和tf_pb及input_names一起提供。 |

#### 2.2 PyTorch 模型特有参数说明

| 参数名称 | 是否必需 | 输入格式 | 参数描述 | 注意事项 |
|---|---|---|---|---|
| pytorch_pt            | 是 | --pytorch_pt path/to/file         | PyTorch框架模型文件路径 | 1. 需将框架名指定为pytorch。2. 仅支持pytorch1.6 JIT导出的模型。 |
| pt_input_dtypes       | 否 | --pt_input_dtypes dtype1  dtype2 ...| PyTorch模型输入的数据类型 | 1. 需将框架名指定为pytorch。2. 默认值为FLOAT（单输入）。3. 多输入时，--pt_input_dtypes后依次跟每个输入的数据类型，各个数据类型之间以空格分隔。 |

#### 2.3 Caffe 模型特有参数说明

| 参数名称 | 是否必需 | 输入格式 | 参数描述 | 注意事项 |
|---|---|---|---|---|
| prototxt              | 是 | --prototxt path/to/file           | Caffe模型图结构文件路径 | 1. 需要和caffemodel一起提供。2. 需将框架名指定为caffe。 |
| caffemodel            | 是 | --caffemodel path/to/file         | Caffe模型权值文件路径 | 1. 需要和prototxt一起提供。2. 需将框架名指定为caffe。 |

#### 2.4 Onnx 模型特有参数说明

| 参数名称 | 是否必需 | 输入格式 | 参数描述 | 注意事项 |
|---|---|---|---|---|
| onnx_model                  | 是 | --onnx_model path/to/file               | ONNX框架模型文件路径 | 需将框架名指定为onnx。 |


### common_calibrate.py 详细说明

该脚本接收传入的args, network, config以及根据自定义的CalibData类生成的calibdata，在脚本内生成calibrator并完成calibrate操作。

### framework_parser.py 详细说明

该脚本内包含了4种框架的parser类（TensorFlowParser，PytorchParser，CaffeParser，OnnxParser），实际使用时通过实例化所需框架的类后，再调用parse()方法即可完成第三方框架模型的解析。

###  logger.py 详细说明
日志打印公共脚本，支持DEBUG, INFO, WARNING, ERROR 4种粒度的日志输出，输出示例如下：
```bash
2023-03-24 10:01:12,864: INFO: infer.py:67] Device count:1
2023-03-24 10:01:13,080: INFO: infer.py:91] Start run ...
```
###  model_process.py 详细说明

该脚本内包含了MagicMind模型参数提取、编译、配置、序列化所需的公共函数。通常您将用到的是extract_params(), config_network(), get_builder_config(), build_and_serialize()。其余函数均可理解为内部函数。

## 参考文档：
magicmind发布包内的 magicmind/python/samples/mm_build/README.md

