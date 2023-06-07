import magicmind.python.runtime as mm
import numpy as np
import cv2
from typing import List
from logger import Logger

log = Logger()


class CalibData(mm.CalibDataInterface):
    def __init__(
        self, shape: mm.Dims, max_samples: int, resize_h, resize_w, video_list: List
    ):
        super().__init__()
        self.captures_ = []
        for path in video_list:
            self.captures_.append(cv2.VideoCapture(path.strip()))
        self.shape_ = shape
        self.max_samples_ = max_samples
        self.cur_capture_index_ = 0
        self.processed_clips_ = 0
        self.resize_w = resize_w
        self.resize_h = resize_h

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_

    # read a clip
    def read_clip(self):
        clip_len = self.shape_.GetDimValue(2)
        while self.cur_capture_index_ < len(self.captures_):
            capture = self.captures_[self.cur_capture_index_]
            clip = []
            while len(clip) < clip_len:
                got_frame, frame = capture.read()
                if not got_frame:
                    break
                else:
                    clip.append(frame)
            if len(clip) < clip_len:
                # reach the end of the video, read next video
                self.cur_capture_index_ = self.cur_capture_index_ + 1
            else:
                return clip
        log.info("get None")
        # no more data
        return None

    def next(self):
        if self.processed_clips_ >= self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "Maximum samples reached")
        batch_size = self.shape_.GetDimValue(0)
        clip_len = self.shape_.GetDimValue(2)
        h = self.shape_.GetDimValue(3)
        w = self.shape_.GetDimValue(4)
        clips = []
        for i in range(batch_size):
            clip = self.read_clip()
            if clip is None:
                # no more data
                return mm.Status(mm.Code.OUT_OF_RANGE, "Data end reached")
            clips.append(clip)

        # resize and center crop
        preprocessed_clips = []
        for clip in clips:
            preprocessed_clip = []
            for frame in clip:
                resized = cv2.resize(frame, (self.resize_w, self.resize_h))
                top = round((self.resize_h - h) / 2)
                left = round((self.resize_w - w) / 2)
                cropped = resized[top : top + h, left : left + w, :]
                normalized = cropped - [128, 128, 128]
                preprocessed_clip.append(normalized)
            preprocessed_clip = np.array(preprocessed_clip)  # DHWC
            preprocessed_clip = preprocessed_clip.transpose((3, 0, 1, 2))  # CDHW
            preprocessed_clips.append(preprocessed_clip)
        self.cur_sample_ = np.array(preprocessed_clips).astype("float32")  # NCDHW
        self.processed_clips_ = self.processed_clips_ + 1
        return mm.Status.OK()

    def reset(self):
        self.cur_sample_ = None
        self.cur_capture_index_ = 0
        self.processed_clips_ = 0
        return mm.Status.OK()
