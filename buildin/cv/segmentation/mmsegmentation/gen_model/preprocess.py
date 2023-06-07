import cv2
import numpy as np
import os 

img_size = os.environ.get("MMSEGMENTATION_MODEL_IMAGE_SIZE")
img_size = img_size.split(',')
h,w = int(img_size[0]),int(img_size[1])

def imrescale(img, scale):
    h, w = img.shape[:2]
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                        max_short_edge / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    resized_img = cv2.resize(
            img, new_size, dst=None, interpolation=cv2.INTER_LINEAR)
    new_h, new_w = resized_img.shape[:2]
    w_scale = new_w / w
    h_scale = new_h / h
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale])
    return resized_img, scale_factor

def imnormalize(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.
    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    img = img.astype(np.float32)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    img = img.transpose(2,0,1)
    return np.expand_dims(img, axis=0)

def preprocess(image):
    scale = (w, h)
    mean = np.array([123.675, 116.28 , 103.53])
    std = np.array([58.395, 57.12 , 57.375])
    resized_img, scale_size = imrescale(image, scale)
    img = imnormalize(resized_img, mean, std)
    return img, scale_size