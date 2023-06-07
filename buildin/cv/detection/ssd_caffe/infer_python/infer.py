# from __future__ import print_function
import argparse
import os,sys
import numpy as np
import magicmind.python.runtime as mm
import cv2

_classes = ('__background__',
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='mAP Calculation')
    parser.add_argument('--magicmind_model', type=str, required=True)
    parser.add_argument('--result_path', help='The result data path', type=str)
    parser.add_argument('--devkit_path', help='VOCdevkit path', type=str)
    parser.add_argument('--save_img', type=bool, default=False)
    args = parser.parse_args()

    return args

def get_results(path, pred_dir, save_img):
    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)
    threshold = 0.01
    input_size = [300, 300]

    def preprocess_images(image_path: str, dst_size: list):
        imgs = []
        dst_h, dst_w = dst_size[0], dst_size[1]
        ori = cv2.imread(image_path)
        ori_shape = ori.shape
        scale = [float(ori_shape[0]) / float(dst_h), float(ori_shape[1]) / float(dst_w)]
        img = cv2.resize(ori, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        # mean std
        img = img.astype(np.float32)
        img -= 127.5
        img *= 0.007843
        imgs.append(np.ascontiguousarray(img)[np.newaxis,:])
        # batch
        return np.ascontiguousarray(np.concatenate(tuple(imgs), axis=0)), ori, scale

    # 读取测试图片文件名
    filenames = []
    with open(os.path.join(path,'VOC2007/ImageSets/Main/test.txt'), 'r') as f:
        filenames = f.readlines()
    # 生成图片列表文件
    images_path = [os.path.join(path, 'VOC2007/JPEGImages/' + filename.strip() + '.jpg')\
              for filename in filenames]
    #创建保存结果的文件夹

    voc_preds_files = []
    folder = os.path.exists(pred_dir)
    if not folder: 
        os.makedirs(pred_dir)
    for t in _classes:
        voc_preds_files.append(open(pred_dir + '/comp3_det_test_' + t + '.txt', 'w'))
    total_num = images_path.__len__()
    print('total image number:{}'.format(total_num))
    with mm.System():
        dev = mm.Device()
        dev.id = 0
        econfig = mm.Model.EngineConfig()
        econfig.device_type = "MLU"
        engine = model.create_i_engine(econfig)
        assert engine != None, "Failed to create engine"
        context = engine.create_i_context()
        queue = dev.create_queue()
        assert queue != None

        inputs = context.create_inputs()
        for i, path in enumerate(images_path):
            print('{}/{}:{}'.format(i+1, total_num, path))
            img, ori, scale = preprocess_images(path, input_size)
            inputs[0].from_numpy(img)
            inputs[0].to(dev)
            outputs = []
            assert context.enqueue(inputs, outputs, queue).ok()
            assert queue.sync().ok()

            results = outputs[0].asnumpy()
            out_shape = results.shape
            bbox_size = 7  # every 7 values form a bounding box.
            assert out_shape[1] == bbox_size

            bbox_num = out_shape[0]
            for bbox in results:
                batch_id = bbox[0]
                category = int(bbox[1])
                # background
                if 0 == category: continue
                
                if (threshold > bbox[2]): continue

                left = float(bbox[3] * float(img.shape[2]) * scale[1])  # left
                top = float(bbox[4] * float(img.shape[1]) * scale[0])  # top
                right = float(bbox[5] * float(img.shape[2]) * scale[1])  # right
                bottom = float(bbox[6] * float(img.shape[1]) * scale[0])  # bottom
                
                if left >= right or top >= bottom : continue
                # check border
                left = max(0, left)
                right = min(right, ori.shape[1])
                top = max(top, 0)
                bottom = min(bottom, ori.shape[0])

                # write filename first
                voc_preds_files[category].write(filenames[i].strip())
                # voc_preds_files[category].write(' ' + str(category))
                voc_preds_files[category].write(' ' + str(bbox[2]))
                voc_preds_files[category].write(' ' + str(left))
                voc_preds_files[category].write(' ' + str(top))
                voc_preds_files[category].write(' ' + str(right))
                voc_preds_files[category].write(' ' + str(bottom))
                voc_preds_files[category].write('\n')

                if save_img:
                    cv2.rectangle(ori, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 255))
                    text = _classes[category] + ": " + str(bbox[2])
                    text_size, _ = cv2.getTextSize(text, 0, 0.5, 1)
                    cv2.putText(ori, text, (int(left), int(top) + text_size[1]), 0, 0.5, (255, 255, 255), 1)
            if save_img:
                cv2.imwrite(os.path.join(pred_dir, filenames[i].strip() + '.jpg'), ori)

    for file in voc_preds_files:
        file.close()


if __name__ == '__main__':
    args = parse_args()
    get_results(args.devkit_path, args.result_path, args.save_img)
