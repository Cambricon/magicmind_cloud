from magicmind.python.runtime.parser import Parser
import magicmind.python.runtime as mm

import argparse
import numpy as np
import glob
import os
import cv2 
# Parameters

def do_calibrate(network, calib_data, config, precision):
    # create calibrator
    print("calibration data : ",calib_data.get_sample)
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
    # original_image_size = original_image.shape[:2]
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
            print(self.data_paths_[i])
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

def construct_model(opt, pb_file, imageset_dir, reload_by_serilize=True):
    # init builder, network, builder_config and parser
    builder = mm.Builder()
    yolov3_network = mm.Network()

    config = mm.BuilderConfig()
    parser = Parser(mm.ModelKind.kTensorflow)

    # get input dims from network  
    input_names = ["input/input_data"]
    output_names = ["conv_sbbox/BiasAdd", "conv_mbbox/BiasAdd", "conv_lbbox/BiasAdd"]

    parser.set_model_param("tf-model-type","tf-graphdef-file")
    parser.set_model_param("tf-graphdef-inputs", input_names)
    parser.set_model_param("tf-graphdef-outputs", output_names)
    assert parser.parse(yolov3_network, pb_file).ok()

    config.parse_from_string('{"archs":[{"mtp_372": [2,6,8]}]}')
    config.parse_from_string('{"opt_config":{"type64to32_conversion": true}}')
    config.parse_from_string('{"opt_config":{"conv_scale_fold": true}}')

    assert yolov3_network.get_input(0).set_dimension(mm.Dims((opt.batch_size,  opt.img_size[0], opt.img_size[1],3))).ok()

    if opt.shape_mutable=='true':
        assert config.parse_from_string('{"graph_shape_mutable": true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 416, 416], "max": [32, 3, 416, 416]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable": false}').ok()
    
    calib_data = FixedCalibData(mm.Dims([opt.batch_size, opt.img_size[0], opt.img_size[1], 3]),
                                max_samples = opt.batch_size,
                                img_dir = imageset_dir)

    conf_thres = 0.3
    iou_thres = 0.45
    class_num = 80
    order = [0,1,2]
    
    output_tensors = []
    for i in order:
        tensor = yolov3_network.get_output(i)
        output_tensors.append(tensor)

    output_count = yolov3_network.get_output_count()
    for i in range(output_count):
        yolov3_network.unmark_output(yolov3_network.get_output(0))
    
    # anchors，按原始3个yolo层顺序填写
    bias_buffer = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326 ]
    bias_node = yolov3_network.add_i_const_node(mm.DataType.FLOAT32, mm.Dims([len(bias_buffer)]),
        np.array(bias_buffer, dtype=np.float32))
    detect_out = yolov3_network.add_i_detection_output_node(output_tensors, bias_node.get_output(0))
    detect_out.set_algo(mm.IDetectionOutputAlgo.YOLOV3)
    detect_out.set_confidence_thresh(conf_thres)
    detect_out.set_nms_thresh(iou_thres)
    detect_out.set_scale(1.0)
    detect_out.set_num_coord(4)
    detect_out.set_num_class(class_num)
    detect_out.set_num_entry(5)
    detect_out.set_num_anchor(3)
    detect_out.set_num_box_limit(1024)
    detect_out.set_image_shape(opt.img_size[0], opt.img_size[1])
    detect_out.set_layout(mm.Layout.NONE, mm.Layout.NONE)
    # 将detect_out层输出标记为网络输出
    detection_output_count = detect_out.get_output_count()
    for i in range(detection_output_count):
        yolov3_network.mark_output(detect_out.get_output(i))

    do_calibrate(yolov3_network, calib_data, config, opt.precision)
    # build model from calibrated network
    mm_model = builder.build_model("yolov3_quanmodel", yolov3_network, config)
    
    assert mm_model.serialize_to_file(opt.mm_model).ok()
    print(opt.mm_model," was saved successfully in the current path.")

def main(opt):
    construct_model(opt, opt.tf_pb, opt.datasets_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_pb', type=str, default='../data/yolov3_coco_mmpost.pb', help='tf_pb path(s)')
    parser.add_argument('--mm_model', type=str, default='../data/', help='saved .mm model name')
    parser.add_argument('--precision', type=str, default='force_float32', help='precision')
    parser.add_argument('--shape_mutable', type=str, default="false", required=True ,help='shape_mutable')
    parser.add_argument('--datasets_dir', type=str, default='../data/COCO/images/', help='quantized data path,default: /data/datasets/COCO2017/images/val2017')
    parser.add_argument('--quan_img_num', type=int, default=5, help='quantized img num, default:10')
    parser.add_argument('--img_size', type=list, default=[416, 416], help='inference size (pixels)')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch, default:1') 
    opt = parser.parse_args()
    main(opt)

