from magicmind.python.runtime.parser import Parser
import magicmind.python.runtime as mm

import argparse
import numpy as np
import glob
import os
import cv2 
# Parameters

def do_calibrate(network, calib_data, config, precision):
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    assert config.parse_from_string(
        """{"precision_config": {"precision_mode": "%s"}}"""%(precision)).ok()
    # calibrate the network
    assert calibrator.calibrate(network, config).ok()

def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    ih, iw    = target_size
    h,  w, _  = image.shape
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.
    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def preprocess_image(img, input_size) -> np.ndarray:

    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_data = image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data.astype(np.float32)    
    return image_data
class FixedCalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        assert os.path.isdir(img_dir)
        self.data_paths_ = glob.glob(img_dir + '/*.jpg')
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.dst_shape_ = (self.shape_.GetDimValue(2), self.shape_.GetDimValue(3))

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_
    
    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        for i in range(data_begin, data_end):
            img = cv2.imread(self.data_paths_[i])
            img = preprocess_image(img, self.dst_shape_[0])
            imgs.append(img[np.newaxis,:])
        # batch and normalize
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0))

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        data_begin = self.cur_data_index_
        data_end = data_begin + batch_size
        if data_end > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
        self.cur_sample_ = self.preprocess_images(data_begin, data_end)
        self.cur_data_index_ = data_end
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        return mm.Status.OK()

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

def construct_model(opt, caffemodel, prototxt, imageset_dir, reload_by_serilize=True):
    # init builder, network, builder_config and parser
    builder = mm.Builder()
    yolov4_network = mm.Network()

    config = mm.BuilderConfig()
    parser = Parser(mm.ModelKind.kCaffe)
   
    assert parser.parse(yolov4_network, caffemodel, prototxt).ok()

    config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}')
    config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}')
    config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}')
    assert config.parse_from_string('{"convert_input_layout": { "0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    
    assert yolov4_network.get_input(0).set_dimension(mm.Dims((opt.batch_size, 3, opt.img_size[0], opt.img_size[1]))).ok()

    if opt.shape_mutable=='true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 416, 416], "max": [32, 3, 416, 416]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    
    calib_data = FixedCalibData(mm.Dims([opt.batch_size, 3,opt.img_size[0], opt.img_size[1]]),
                                max_samples = opt.batch_size,
                                img_dir = imageset_dir)

    conf_thres = 0.3
    iou_thres = 0.45
    class_num = 80
    perms = [0, 2, 3, 1]  # 0 : N, 1 : C, 2 : H, 3 : W
    const_node = yolov4_network.add_i_const_node(mm.DataType.INT32, mm.Dims([len(perms)]), np.array(perms, dtype=np.int32))
    output_tensors = []
    for i in range(yolov4_network.get_output_count()):
        # 添加premute算子做NCHW到NHWC的转换
        tensor = yolov4_network.get_output(i)
        permute_node = yolov4_network.add_i_permute_node(tensor, const_node.get_output(0))
        output_tensors.append(permute_node.get_output(0))
    output_count = yolov4_network.get_output_count()
    for i in range(output_count):
        # 去掉原网络输出tensor标志
        yolov4_network.unmark_output(yolov4_network.get_output(0))

    bias_buffer = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401 ]
    bias_node = yolov4_network.add_i_const_node(mm.DataType.FLOAT32, mm.Dims([len(bias_buffer)]),
        np.array(bias_buffer, dtype=np.float32))
    detect_out = yolov4_network.add_i_detection_output_node(output_tensors, bias_node.get_output(0))
    detect_out.set_algo(mm.IDetectionOutputAlgo.YOLOV4)
    detect_out.set_confidence_thresh(conf_thres)
    detect_out.set_nms_thresh(iou_thres)
    detect_out.set_scale(1)
    detect_out.set_num_coord(4)
    detect_out.set_num_class(class_num)
    detect_out.set_num_entry(5)
    detect_out.set_num_anchor(3)
    detect_out.set_aspect_ratios([1.2,1.1,1.05])  # scale_x_y
    detect_out.set_num_box_limit(1024)
    detect_out.set_image_shape(opt.img_size[0], opt.img_size[1])
    detect_out.set_layout(mm.Layout.NONE, mm.Layout.NONE)
    # 将detect_out层输出标记为网络输出
    detection_output_count = detect_out.get_output_count()
    for i in range(detection_output_count):
        yolov4_network.mark_output(detect_out.get_output(i))

    do_calibrate(yolov4_network, calib_data, config, opt.precision)
    # build model from calibrated network
    mm_model = builder.build_model("yolov4_quanmodel", yolov4_network, config)
    assert mm_model.serialize_to_file(opt.mm_model).ok()
    print(opt.mm_model," was saved successfully in the current path.")

def main(opt):
    construct_model(opt, opt.caffemodel, opt.prototxt, opt.datasets_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--caffemodel', type=str, default='../data/yolov4.caffemodel', help='caffemodel path(s)')
    parser.add_argument('--prototxt', type=str, default='../data/yolov4.prototxt', help='prototxt path(s)')
    parser.add_argument('--mm_model', type=str, default='../data/', help='saved .mm model name')
    parser.add_argument('--precision', type=str, default='force_float32', help='precision')
    parser.add_argument('--shape_mutable', type=str, default="false", required=True ,help='shape_mutable')
    parser.add_argument('--datasets_dir', type=str, default='../data/COCO/images/', help='quantized data path,default: /data/datasets/COCO2017/images/val2017')
    parser.add_argument('--quan_img_num', type=int, default=5, help='quantized img num, default:10')
    parser.add_argument('--img_size', type=list, default=[416, 416], help='inference size (pixels)')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch, default:1') 
    opt = parser.parse_args()
    main(opt)

