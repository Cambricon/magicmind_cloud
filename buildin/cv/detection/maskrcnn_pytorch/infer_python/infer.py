from cgi import print_form
import magicmind.python.runtime as mm
import argparse
import numpy as np
import cv2
import os
from tqdm import tqdm
from pre_process import Record,load_images,preprocess_img
from post_process import apply_mask,random_colors

parser = argparse.ArgumentParser()
parser.add_argument("--magicmind_model", "--magicmind_model",type=str,default="")
parser.add_argument("--device_id", "--device_id", type = int, default = 0, help = "device_id")
parser.add_argument("--image_dir", "--image_dir",  type=str, default="", help="coco val datasets")
parser.add_argument("--label_dir", "--label_dir",  type=str, default="", help="coco names dir")
parser.add_argument("--output_img_dir", "--output_img_dir", type=str, default="")
parser.add_argument("--output_maskimg_dir", "--output_maskimg_dir", type=str, default="")
parser.add_argument("--output_pred_dir", "--output_pred_dir", type=str, default="")
parser.add_argument("--save_imgname_dir", "--save_imgname_dir", type=str, default="")
parser.add_argument("--save_img", "--save_img", type=int, default=1)
parser.add_argument("--save_mask", "--show_mask", type=int, default=1)
parser.add_argument("--score_th", "--score_th", type = float, default = 0.3, help = "score_threshold")
parser.add_argument('--input_size', type=int, default=800, required=True ,help='input_size')
parser.add_argument('--test_nums', type=int, default=-1, required=True ,help='test_nums')

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)
    
    img_size = [3,args.input_size,args.input_size]
    batch_size = 1 #当前仅支持batch_size=1 多batch情况下外加一层batch循环即可 保留传入参数batch_size
    
    # load images 循坏补齐为batch_size的整数倍
    images_list = load_images(args.image_dir,batch_size)
    
    # load name_dict
    name_dict = np.loadtxt(args.label_dir, dtype='str', delimiter='\n')
    
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        assert args.device_id < dev_count
        # 打开MLU设备
        dev = mm.Device()
        dev.id = args.device_id
        assert dev.active().ok()
        # 创建Engine
        econfig = mm.Model.EngineConfig()
        econfig.device_type = "MLU"
        engine = model.create_i_engine(econfig)
        assert engine != None, "Failed to create engine"
        # 创建Context
        context = engine.create_i_context()
        assert context != None
        # 创建MLU任务队列
        queue = dev.create_queue()
        assert queue != None
        # 创建输入tensor, 输出tensor
        inputs = context.create_inputs()
        # [0]:bsx100x5xsizeof(float) [1]:bsx100xsizeof(float) [2]:bsx100x224x224xsizeof(float)
        outputs = []
        
        total_images = len(images_list)
        np_input = np.zeros((batch_size,*img_size)).astype(np.float32)
        N = 80 # class nums
        colors = random_colors(N)
        with open(os.path.join(args.save_imgname_dir,"image_name.txt"),'w') as imgname_f:
            # start inference
            total_images = min(args.test_nums,total_images) if args.test_nums != -1 else total_images
            for i in tqdm(range(total_images)):
                # pre-process
                image_name  = images_list[i]
                src_img = cv2.imread(os.path.join(args.image_dir,images_list[i]))
                masked_image = src_img.copy()
                np_input[ i % batch_size,:,:,:],ratio= preprocess_img(src_img,img_size)
                # post-process including bbox and segmask
                inputs[0].from_numpy(np_input)
                assert context.enqueue(inputs, outputs, queue).ok()
                assert queue.sync().ok()
                output_bboxes,output_scores,output_segmask = outputs[0].asnumpy(),outputs[1].asnumpy(),outputs[2].asnumpy()
                record = Record(args.output_pred_dir + "/" + image_name.replace('.jpg','.txt').replace('.JPEG','.txt'))
                rois = output_bboxes[0] # 100x5 (idx, x1, y1, x2, y2)
                rois_label = output_scores[0] # 100x1 pred_class_id
                rois_segmask = output_segmask[0]
                roi_nums = len(rois)
                scale_w = 1.0 * src_img.shape[1] / img_size[2]
                scale_h = 1.0 * src_img.shape[0] / img_size[1]
                for _roi_idx in range(roi_nums):
                    # post-process
                    x1,y1,x2,y2,score = (rois[_roi_idx])
                    if score > args.score_th:
                        # process bbox
                        xmin = int(max(0,min(min(x1,x2),img_size[2]))*scale_w)
                        xmax = int(max(0,min(max(x1,x2),img_size[2]))*scale_w)
                        ymin = int(max(0,min(min(y1,y2),img_size[1]))*scale_h)
                        ymax = int(max(0,min(max(y1,y2),img_size[1]))*scale_h)
                        label = int(rois_label[_roi_idx])
                        result = name_dict[label]+"," +str(score)+","+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)
                        record.write(result, False)
                        if args.save_img:
                            cv2.rectangle(src_img, (xmin, ymin), (xmax, ymax), (0, 255, 0))
                        text = name_dict[label] + ": " + str(score)
                        text_size, _ = cv2.getTextSize(text, 0, 0.5, 1)
                        cv2.putText(src_img, text, (xmin, ymin + text_size[1]), 0, 0.5, (255, 255, 255), 1)
                        # process segmmask
                        if args.save_mask:
                            mask = rois_segmask[_roi_idx]
                            mask = cv2.resize(mask,(src_img.shape[1],src_img.shape[0]))
                            masked_image = apply_mask(masked_image, mask, colors[label])
                            cv2.imwrite(args.output_maskimg_dir + "/" + image_name, masked_image)
                imgname_f.writelines(os.path.join(args.image_dir,images_list[i])+'\n')
                if args.save_img:
                    cv2.imwrite(args.output_img_dir + "/" + image_name, src_img)
            imgname_f.close()