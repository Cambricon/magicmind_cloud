import magicmind.python.runtime as mm
import math
import argparse
import numpy as np
import torch
from torchvision import transforms#, utils
import glob
import os
from skimage import transform, io
from magicmind.python.common.types import get_numpy_dtype_by_datatype
import cv2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--device_id', dest = 'device_id', default = 0,
                    type = int, help = 'mlu device id')
parser.add_argument("--magicmind_model", type=str, default="magicmind_model",
                    help="magicmind_model")
parser.add_argument("--img_dir", type=str, help="images directory",
                    required=True)
parser.add_argument("--output_folder", type=str, default="../data/output",
                    help="forder for saving results.")
parser.add_argument('--batch_size', dest = 'batch_size', default = 1,
                    type = int, help = 'batch_size')
parser.add_argument('--save_img', dest = 'save_img', default = False,
                    type = bool, help = 'save result in images')

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

transform1 = transforms.Compose([
	transforms.ToTensor(), 
	]
)

def eval_mae(y_pred, y):
    return torch.abs(y_pred - y).mean()

beta = math.sqrt(0.3)  # for max F_beta metric
# get precisions and recalls: threshold---divided [0, 1] to num values
def eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num), torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
    return prec, recall

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    imo.save(d_dir+imidx+'.png')

def preprocess(img):
    output_size = (320, 320)
    new_h, new_w = output_size
    image = transform.resize(img,(new_h, new_w),mode='constant')
    tmpImg = np.zeros((image.shape[0], image.shape[1],3))
    image = image/np.max(image)
    if image.shape[2] == 1:
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
    else:
        tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
        tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
        tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
    tmpImg = tmpImg.transpose((2, 0, 1))
    return tmpImg

def load_processed_image(file_path):
    image = io.imread(file_path)
    tmpImg = preprocess(image)
    return tmpImg

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)

    with mm.System():
        dev = mm.Device()
        dev.id = args.device_id
        assert dev.active().ok()
        econfig = mm.Model.EngineConfig()
        econfig.device_type = "MLU"
        engine = model.create_i_engine(econfig)
        assert engine != None, "Failed to create engine"
        context = engine.create_i_context()
        queue = dev.create_queue()
        assert queue != None

        inputs = context.create_inputs()
        outputs = []

        img_name_list = glob.glob(args.img_dir + os.sep + '*.jpg')
        label_name_list = glob.glob(args.img_dir + os.sep + '*.png')
        img_name_list.sort()
        label_name_list.sort()
        image_num = len(img_name_list)
        pad_num = args.batch_size - (image_num % args.batch_size)
        if pad_num != args.batch_size:
            for i in range(pad_num):
                img_name_list.append(img_name_list[0])
            image_num += pad_num
        loop_num = 0
        avg_mae = 0.0
        num= 100
        avg_prec, avg_recall = torch.zeros(num), torch.zeros(num)
        while loop_num < image_num:
            images = []
            image_names = []
            for i in range(args.batch_size):
                images.append(load_processed_image(img_name_list[loop_num + i])[np.newaxis, :])
                image_names.append(img_name_list[loop_num+ i])
            _input_data = np.concatenate(tuple(images))
            input_data = np.ascontiguousarray(_input_data.astype(dtype=get_numpy_dtype_by_datatype(mm.DataType.FLOAT32)))
            inputs[0].from_numpy(input_data)
            status =context.enqueue(inputs, outputs, queue)
            queue.sync()

            d1 = torch.from_numpy(outputs[0].asnumpy())
            for i in range(args.batch_size):
                labels = cv2.imread(label_name_list[loop_num + i])
                labels = labels[:,:,0]
                pred = d1[i:i+1,0,:,:]
                pred_ori = normPRED(pred)
                if args.save_img:
                    save_output(image_names[i], pred_ori, args.output_folder)
                pred_s = pred_ori.squeeze()
                pred_np = pred_s.data.numpy()
                predict_np = Image.fromarray(pred_np * 255)
                imo = predict_np.resize((labels.shape[1],labels.shape[0]),resample=Image.BILINEAR)
                imo = transform1(imo)
                labels = torch.tensor(labels)
                labels = labels.unsqueeze(0)
                imo = imo.permute(1,2,0)
                labels = labels.permute(1,2,0)
                mae = eval_mae(imo, labels)
                prec, recall = eval_pr(imo, labels, num)
                avg_mae += mae
                avg_prec, avg_recall = avg_prec + prec, avg_recall + recall

            loop_num += args.batch_size

        avg_mae, avg_prec, avg_recall = avg_mae / image_num, avg_prec / image_num, avg_recall / image_num
        score = (1 + beta ** 2) * avg_prec * avg_recall / (beta ** 2 * avg_prec + avg_recall)
        score[score != score] = 0  # delete the nan
        print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()))
        f = open(args.output_folder+os.sep+"result.txt", "w")
        f.write('average mae: %.4f\n' %(avg_mae))
        f.write('max fmeasure: %.4f\n' %(score.max()))
        f.close()


