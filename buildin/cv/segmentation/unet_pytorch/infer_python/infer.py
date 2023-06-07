import magicmind.python.runtime as mm
import argparse
import numpy as np
import torch
import os
import sys
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from PIL import Image
from data_loading import BasicDataset, CarvanaDataset
from torch.utils.data import DataLoader, random_split

from mm_runner import MMRunner
from logger import Logger

log = Logger()
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 959



MODEL_PATH = os.getenv("MODEL_PATH")
CARVANA_DATASETS_PATH = os.getenv("CARVANA_DATASETS_PATH")
PROJ_ROOT_PATH = os.getenv("PROJ_ROOT_PATH")

parser = argparse.ArgumentParser()
parser.add_argument(
        "--device_id", dest="device_id", type=int, default=0, help="device_id"
)
parser.add_argument(
        "--magicmind_model", dest="magicmind_model",type=str, default=str(MODEL_PATH) + "/unet_carvana_scale0.5_epoch2.mm", help="magicmind_model"
)
parser.add_argument(
        "--data_folder",  dest="data_folder",type=str, default = str(CARVANA_DATASETS_PATH)
)
parser.add_argument(
        "--output_folder",  dest="output_folder",type=str, default=  "../data/output",
                    help="forder for saving output."
)
parser.add_argument(
        "--classes", "-c",  dest="classes",type=int, default=2, help="Number of classes"
)
parser.add_argument(
        "--scale", "-s",  dest="scale",type=float, default=0.5, help="Downscaling factor of the images"
)
parser.add_argument(
        "--batch_size", "-b", dest="batch_size", metavar="B", type=int, default=1, help="Batch size"
)
parser.add_argument(
        "--validation", "-v", dest="val", type=float, default=10.0,
                        help="Percent of the data that is used as validation (0-100)"
)
parser.add_argument(
        "--save_img",  dest="save_img",action="store_true", default=False , help="Save the output masks"
)
parser.add_argument(
        "--mask-threshold","-t", dest="mask-threshold", type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white'
)
parser.add_argument(
    "--image_num", dest="image_num", type=int, default=10, help="image number"
)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

class MM_Model(object):
    def __init__(self, model_path,device_id):
        #mlu
        self.model = MMRunner(mm_file=model_path, device_id=device_id)
    def forward(self, input):
        outputs = []
        inputs = [input]
        outputs = self.model(inputs)

        return outputs[0]

def main():
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()

    Unet = MM_Model(args.magicmind_model,args.device_id)
    dir_img = args.data_folder + '/imgs/'
    dir_mask = args.data_folder + '/masks/'
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, args.scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, args.scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * args.val / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers= 0, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    num_val_batches = len(val_loader)
    dice_score = 0
    number = 0
    count = 0
    log.info("Start run ...")
    image_num = args.image_num
    #image_num = num_val_batches
    batch_size=args.batch_size
    if image_num >= 508:
        image_num = 508
    rem_img_num = image_num % batch_size
    img_idx = 0
    batch_counter = 0
    total_tmp = int(image_num/batch_size)
    imgs = np.empty([batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH])
    if total_tmp*batch_size < image_num:
        total_tmp += 1
    for batch in tqdm(val_loader, total=total_tmp, desc='Validation round', unit='batch', leave=False):
        image_ori, mask_true = batch['image'], batch['mask']
        infer_batch_size = (
            batch_size if img_idx < (image_num - rem_img_num) else rem_img_num
        )
        if infer_batch_size == 0:
            break
        # predict the mask
        #image = image_ori.permute(0,2,3,1)
        mask_trues = torch.split(mask_true,1,dim=0)
        for index in range (0,infer_batch_size):
            image = image_ori[index]
            imgs[batch_counter % infer_batch_size, :, :, :] = image
            batch_counter += 1
            img_idx += 1
        if batch_counter % infer_batch_size == 0:
            batch_counter = 0
            output = Unet.forward(imgs)
            mask_pred = torch.tensor(output.astype(np.float32))
            mask_preds = torch.split(mask_pred,1,dim=0)
            for pred_idx in range(infer_batch_size):
                if args.save_img:
                    out_filename = str(number) + '.jpg'
                    mask_values = [0, 1]
                    pad_h = int(image_ori.shape[3] / args.scale)
                    pad_w = int(image_ori.shape[2] / args.scale)
                    output = F.interpolate(mask_pred, (pad_w, pad_h), mode='bilinear')
                    if args.classes > 1:
                        mask = output.argmax(dim=1)
                    else:
                        mask = torch.sigmoid(output) > args.mask_threshold
                    make_return = mask[0].long().squeeze().numpy()
                    result = mask_to_image(make_return, mask_values)
                    result.save(args.output_folder + os.sep + out_filename)
                    number = number + 1
                    print('Mask saved to ' + args.output_folder + os.sep + out_filename)

                if args.classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < args.classes, 'True mask indices should be in [0, classes['
                    # convert to one-hot format
                    mask_true1 = F.one_hot(mask_trues[pred_idx], args.classes).permute(0, 3, 1, 2).float()
                    mask_pred1 = F.one_hot(mask_preds[pred_idx].argmax(dim=1), args.classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred1[:, 1:], mask_true1[:, 1:], reduce_batch_first=False)
                count += 1
    val_score =  dice_score / max(image_num, 1)
    print("Dice coefficient:",val_score)
    f = open(args.output_folder + os.sep + "result.txt", "w")
    f.write('Dice coefficient: %.4f\n' %(val_score))
    f.close()


if __name__ == "__main__":
    main()
