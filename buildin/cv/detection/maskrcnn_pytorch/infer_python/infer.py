import sys 
sys.path.append('../export_model/mmdetection')

import argparse
import warnings

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel

from mmdet.apis import single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import compat_cfg

from mmdet.core.export.model_wrappers import DeployBaseDetector
import magicmind.python.runtime as mm
class MagicMindDetector(DeployBaseDetector):
    """ Wrapper for detector's inference with MagicMind."""

    def __init__(self, model_path, class_names, device_id):
        super(MagicMindDetector, self).__init__(class_names, device_id)
        self.mmsys = mm.System()
        assert self.mmsys.initialize().ok()
        self.dev = mm.Device()
        self.dev.dev_id = device_id
        self.dev.active()
        self.queue = self.dev.create_queue()
        self.model = mm.Model()
        self.model.deserialize_from_file(model_path)
        self.engine = self.model.create_i_engine()
        self.context = self.engine.create_i_context()
        self.inputs = self.context.create_inputs()

    def __del__(self):
        for t in self.inputs:
            del t
        del self.context
        del self.engine
        del self.model
        del self.queue
        del self.dev
        self.mmsys.destory()

    def forward_test(self, imgs, img_metas, **kwargs):
        input_data = imgs[0]
        inputs = self.inputs
        assert inputs[0].from_numpy(input_data.numpy()).ok()
        outputs = []
        assert self.context.enqueue(inputs, outputs, self.queue).ok()
        assert self.queue.sync().ok()
        return [x.asnumpy() for x in outputs]

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) an ONNX model using ONNXRuntime')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='Input model file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    
    parser.add_argument("--device_id", "--device_id", type = int, default = 0, help = "device_id")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=2,
        dist=False,
        shuffle=False)

    model = MagicMindDetector(
        args.model, class_names=dataset.CLASSES, device_id=args.device_id)

    model = MMDataParallel(model, device_ids=[args.device_id])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                              args.show_score_thr)

    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == '__main__':
    main()
