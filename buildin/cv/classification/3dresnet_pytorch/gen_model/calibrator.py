import numpy as np
import magicmind.python.runtime as mm
import cv2
import os
import glob
import math
import logging
from PIL import Image
from torchvision.transforms import transforms
import copy
import functools
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
import json
from pathlib import Path
import torch
import torch.utils.data as data
from logger import Logger

log = Logger()


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data["labels"]:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data["database"].items():
        this_subset = value["subset"]
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value["annotations"])
            if "video_path" in value:
                video_paths.append(Path(value["video_path"]))
            else:
                label = value["annotations"]["label"]
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


class VideoDataset(data.Dataset):
    def __init__(
        self,
        root_path,
        annotation_path,
        subset,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
        video_loader=None,
        video_path_formatter=(
            lambda root_path, label, video_id: root_path / label / video_id
        ),
        image_name_formatter=lambda x: f"image_{x:05d}.jpg",
        target_type="label",
    ):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter
        )

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset, video_path_formatter):
        with annotation_path.open("r") as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter
        )
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print("dataset loading [{}/{}]".format(i, len(video_ids)))

            if "label" in annotations[i]:
                label = annotations[i]["label"]
                label_id = class_to_idx[label]
            else:
                label = "test"
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]["segment"]
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                "video": video_path,
                "segment": segment,
                "frame_indices": frame_indices,
                "video_id": video_ids[i],
                "label": label_id,
            }
            dataset.append(sample)
        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __getitem__(self, index):
        path = self.data[index]["video"]
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]["frame_indices"]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


class VideoDatasetMultiClips(VideoDataset):
    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            segments.append([min(clip_frame_indices), max(clip_frame_indices) + 1])

        return clips, segments

    def __getitem__(self, index):
        path = self.data[index]["video"]

        video_frame_indices = self.data[index]["frame_indices"]
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)

        clips, segments = self.__loading(path, video_frame_indices)

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        if "segment" in self.target_type:
            if isinstance(self.target_type, list):
                segment_index = self.target_type.index("segment")
                targets = []
                for s in segments:
                    targets.append(copy.deepcopy(target))
                    targets[-1][segment_index] = s
            else:
                targets = segments
        else:
            targets = [target for _ in range(len(segments))]
        return clips, targets


class Compose(transforms.Compose):
    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):
    def randomize_parameters(self):
        pass


class Normalize(transforms.Normalize):
    def randomize_parameters(self):
        pass


class ScaleValue(object):
    def __init__(self, s):
        self.s = s

    def __call__(self, tensor):
        tensor *= self.s
        return tensor

    def randomize_parameters(self):
        pass


class Resize(transforms.Resize):
    def randomize_parameters(self):
        pass


class Scale(transforms.Scale):
    def randomize_parameters(self):
        pass


class CenterCrop(transforms.CenterCrop):
    def randomize_parameters(self):
        pass


class LoopPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class SlidingWindow(object):
    def __init__(self, size, stride=0):
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        for begin_index in frame_indices[:: self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


class ImageLoaderPIL(object):
    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open("rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")


class VideoLoader(object):
    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video


def image_name_formatter(x):
    return f"image_{x:05d}.jpg"


class VideoDatasetMultiClips(VideoDataset):
    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            segments.append([min(clip_frame_indices), max(clip_frame_indices) + 1])

        return clips, segments

    def __getitem__(self, index):
        path = self.data[index]["video"]
        video_frame_indices = self.data[index]["frame_indices"]
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)

        clips, segments = self.__loading(path, video_frame_indices)

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        if "segment" in self.target_type:
            if isinstance(self.target_type, list):
                segment_index = self.target_type.index("segment")
                targets = []
                for s in segments:
                    targets.append(copy.deepcopy(target))
                    targets[-1][segment_index] = s
            else:
                targets = segments
        else:
            targets = [target for _ in range(len(segments))]
        return clips, targets


class TemporalCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = TemporalCompose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices


class CalibData(mm.CalibDataInterface):
    def __init__(self, shape: mm.Dims, max_samples: int, img_dir: str):
        super().__init__()
        print(img_dir)
        assert os.path.isdir(img_dir)
        self.dir_path = img_dir
        self.data_paths_ = glob.glob(
            img_dir + "abseiling/3E7Jib8Yq5M_000118_000128/*.jpg"
        )
        self.shape_ = shape
        self.max_samples_ = min(max_samples, len(self.data_paths_))
        self.cur_sample_ = None
        self.cur_data_index_ = 0
        self.dst_shape_ = (self.shape_.GetDimValue(2), self.shape_.GetDimValue(3))

        inference_crop = "center"
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
        no_mean_norm = False
        no_std_norm = False
        sample_size = 112
        sample_duration = 16
        inference_stride = 16
        value_scale = 1
        inference_subset = "val"

        normalize = get_normalize_method(mean, std, no_mean_norm, no_std_norm)

        spatial_transform = [Resize(sample_size)]
        spatial_transform.append(CenterCrop(sample_size))
        spatial_transform.append(ToTensor())
        spatial_transform.extend([ScaleValue(value_scale), normalize])
        spatial_transform = Compose(spatial_transform)
        temporal_transform = []

        temporal_transform.append(SlidingWindow(sample_duration, inference_stride))

        temporal_transform = TemporalCompose(temporal_transform)

        loader = VideoLoader(image_name_formatter)
        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id
        )

        video_path = Path(self.dir_path)
        annotation_path = Path(self.dir_path + "../../kinetics.json")
        subset = "validation"
        target_transform = None
        self.inference_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter,
            target_type=["video_id", "segment"],
        )

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return mm.DataType.FLOAT32

    def get_sample(self):
        return self.cur_sample_

    def preprocess_images(self, data_begin: int, data_end: int) -> np.ndarray:
        imgs = []
        k = 0
        count_img = 0
        for i in range(data_begin, data_end):
            if i < (len(self.inference_data.__getitem__(0)[0]) - 1):
                img = self.inference_data.__getitem__(0)[0][i].numpy()
                imgs.append(img[np.newaxis, :])
            elif i == (len(self.inference_data.__getitem__(0)[0]) - 1):
                count_img = i + 1
                while count_img < self.max_samples_:
                    for j in range(len(self.inference_data.__getitem__(k)[0])):
                        imgs.append(img[np.newaxis, :])
                        if (j + 1) == len(self.inference_data.__getitem__(k)[0]):
                            if count_img == self.max_samples_:
                                break
                            else:
                                k = k + 1
                        else:
                            if count_img == self.max_samples_:
                                break
                            else:
                                count_img = count_img + 1

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
