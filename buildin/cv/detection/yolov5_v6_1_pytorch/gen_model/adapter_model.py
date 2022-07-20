from magicmind.python.runtime import ModelKind, IDetectionOutputAlgo, Dims, DataType, Layout
import numpy as np

def append_yolov5_detect(network, conf, iou, max_det):

    # 使用permute算子将三个卷积层输出由NCHW(PyTorch模型数据均为NCHW摆放顺序)转为NHWC摆放顺序
    perms = [0, 2, 3, 1]  # 0 : N, 1 : C, 2 : H, 3 : W
    const_node = network.add_i_const_node(DataType.INT32, Dims(
        [len(perms)]), np.array(perms, dtype=np.int32))
    output_tensors = []
    for i in range(network.get_output_count()):
        # 添加premute算子做NCHW到NHWC的转换
        tensor = network.get_output(i)
        permute_node = network.add_i_permute_node(tensor, const_node.get_output(0))
        output_tensors.append(permute_node.get_output(0))
    output_count = network.get_output_count()
    for i in range(output_count):
        # 去掉原网络输出tensor标志
        network.unmark_output(network.get_output(0))
    
    # anchors，按原始3个yolo层顺序填写
    bias_buffer = [10, 13, 16, 30, 33, 23, 30, 61, 62,
        45, 59, 119, 116, 90, 156, 198, 373, 326 ]
    bias_node = network.add_i_const_node(DataType.FLOAT32, Dims([len(bias_buffer)]),
        np.array(bias_buffer, dtype=np.float32))
    detect_out = network.add_i_detection_output_node(output_tensors, bias_node.get_output(0))
    detect_out.set_algo(IDetectionOutputAlgo.YOLOV5)
    detect_out.set_confidence_thresh(conf)
    detect_out.set_nms_thresh(iou)
    detect_out.set_scale(1.0)
    detect_out.set_num_coord(4)
    detect_out.set_num_class(80)
    detect_out.set_num_entry(5)
    detect_out.set_num_anchor(3)
    detect_out.set_num_box_limit(max_det)
    detect_out.set_image_shape(640, 640)
    detect_out.set_layout(Layout.NONE, Layout.NONE)
    # 将detect_out层输出标记为网络输出
    detection_output_count = detect_out.get_output_count()
    for i in range(detection_output_count):
        network.mark_output(detect_out.get_output(i))
    
    return network
