import magicmind.python.runtime as mm
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.augmentations.utils import pad_nd_image
import argparse
import numpy as np
import torch
import os
import cv2
from mmseg.core.evaluation import get_palette
from preprocess import preprocess
from mmseg.datasets import build_dataloader, build_dataset
import mmcv
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--device_id', dest = 'device_id', default = 0,
            type = int, help = 'mlu device id, used for calibration')
parser.add_argument("--magicmind_model", type=str, default="magicmind_model", help="magicmind_model")
parser.add_argument("--config", type=str, default="../export_model/mmsegmentation/configs/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes.py", 
                        help="model config")
parser.add_argument("--data_root", type=str, default="../data/cityscapes", help="dataset dir")
parser.add_argument("--json_file", type=str, default="result.json", help="save eval result")
CLASS=('road', 'sidewalk', 'building', 'wall', 
        'fence', 'pole', 'traffic light', 'traffic sign', 
        'vegetation', 'terrain', 'sky', 'person', 'rider', 
        'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
PALETTE=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], 
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], 
        [0, 0, 230], [119, 11, 32]]

def show_result(img,
                result,
                palette=None,
                opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = img.copy()
        seg = result
        palette = np.array(palette)
        assert palette.shape[0] == 19
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        return img

def single_mlu_test(context,
                    queue,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (mm.Model): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    inputs = context.create_inputs()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    results = []
    for batch_indices, data in zip(loader_indices, data_loader):
        outputs = []
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        ori_h, ori_w = img_metas[0]['ori_shape'][:-1]
        with torch.no_grad():
            inputs[0].from_numpy(img_tensor.numpy())
            context.enqueue(inputs, outputs, queue)
            queue.sync()
            result = outputs[0].asnumpy()
            result = result.reshape(result.shape[2], result.shape[3], 1).astype(np.uint8)
            preds = cv2.resize(result, (ori_w, ori_h), dst=None, interpolation=cv2.INTER_NEAREST)
            # preds = np.expand_dims(preds, axis=0)
            # print(preds.shape)

        if show or out_dir:            
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(preds, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind {} first!!!".format(args.magicmind_model))
        exit()

    with mm.System():
        model = mm.Model()
        model.deserialize_from_file(args.magicmind_model)
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

        with torch.no_grad():
            cfg = mmcv.Config.fromfile(args.config)
            cfg.data.test['data_root']=args.data_root
            dataset = build_dataset(cfg.data.test)
            cfg.data.test.test_mode = True
            test_loader_cfg = {'num_gpus': 1, 'dist': False, 'shuffle': False, 'samples_per_gpu': 1, 'workers_per_gpu': 2}
            eval_kwargs={'imgfile_prefix': '.format_cityscapes'}
            data_loader = build_dataloader(dataset, **test_loader_cfg)
            results = single_mlu_test(context,
                                    queue,
                                    data_loader,
                                    False,
                                    None,
                                    False,
                                    0.5,
                                    pre_eval=True,
                                    format_only=False,
                                    format_args=eval_kwargs)
            eval_kwargs.update(metric='mIoU')
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, args.json_file, indent=4)
