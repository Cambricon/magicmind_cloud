import os
import cv2
import numpy as np
import magicmind.python.runtime as mm

IJB_DATASETS_PATH = os.environ.get("IJB_DATASETS_PATH")

class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, img_dir: str):
        super().__init__()
        with open(img_dir, 'r') as f:
            image_paths = f.readlines()
        self.images = []
        for image_path in image_paths:
            image = cv2.imread(str(IJB_DATASETS_PATH) + '/' + image_path.strip())
            assert image is not None, 'image [' + image_path.strip() + '] not exists!'
            self.images.append(image)
        nimages = len(self.images)
        assert nimages != 0, 'no images in calibrate list[' + img_dir + ']!'

        self.shape_ = shape
        if nimages < self.shape_.GetDimValue(0):
            for i in range(self.shape_.GetDimValue(0) - nimages):
                self.images.append(self.images[0])

        self.cur_sample_ = None
        self.cur_image_index_ = 0
        self.dst_shape_ = (self.shape_.GetDimValue(2), self.shape_.GetDimValue(3))

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_

    def preprocess_image(self):
        if self.cur_image_index_ == len(self.images):
            return None
        h,w = self.dst_shape_
        image = self.images[self.cur_image_index_]
        image = cv2.resize(image, (w, h))
        image = image.astype('float32')
        image -= 127.5
        image /= 127.5
        image = image.astype('float32')
        image = np.transpose(image, (2, 0, 1)) # HWC >>> CHW
        self.cur_image_index_ = self.cur_image_index_ + 1
        return image

    def next(self):
        batch_size = self.shape_.GetDimValue(0)
        preprocessed_images = []
        for i in range(batch_size):
            image = self.preprocess_image()
            if image is None:
                # no more data
                return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
            preprocessed_images.append(image)
        self.cur_sample_ = np.array(preprocessed_images)
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_image_index_ = 0
        return mm.Status.OK()
