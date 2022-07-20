import magicmind.python.runtime as mm
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.augmentations.utils import pad_nd_image
import argparse
import numpy as np
import torch
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.evaluation.evaluator import evaluate_folder
from preprocess import preprocess, compute_steps_for_sliding_window

parser = argparse.ArgumentParser()
parser.add_argument('--device_id', dest = 'device_id', default = 0,
            type = int, help = 'mlu device id, used for calibration')
parser.add_argument("--magicmind_model", type=str, default="magicmind_model", help="magicmind_model")
parser.add_argument("--model_path",
                    type=str,
                    help="nnUNet_trained_models/nnUNet/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1",
                    required=True)
parser.add_argument("--data_folder",
                    type=str,
                    help="nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr",
                    required=True)
parser.add_argument("--output_folder",
                    type=str,
                    default="../data/results",
                    help="forder for saving results.")
parser.add_argument("--ref_folder",
                    type=str,
                    help="nnUNet_raw_data_base/nnUNet_raw_data/Task02_Heart/labelsTr",
                    required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        print("please generate magicmind model first!!!")
        exit()
    model = mm.Model()
    model.deserialize_from_file(args.magicmind_model)

    pad_kwargs = {'constant_values': 0}
    pad_border_mode = 'constant'
    step_size = 0.5
    num_classes = 2
    num_results = 4
    regions_class_order = None
    preprocessor, list_of_lists, output_files, target_spacing, patch_size, inference_apply_nonlin, transpose_forward, transpose_backward, interpolation_order, force_separate_z, interpolation_order_z= preprocess(args)
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

        with torch.no_grad():
            for i, file_list in enumerate(list_of_lists):
                output_filename = output_files[i]
                d, s, dct = preprocessor.preprocess_test_case(file_list, target_spacing)
                # _internal_predict_3D_2Dconv_tiled
                softmaxes = []
                predicted_segmentation = []
                softmax_pred = []
                for s in range(d.shape[1]):
                    # _internal_predict_2D_2Dconv_tiled
                    assert len(d[:, s].shape) == 3, "x must be (c, x, y)"
                    data, slicer = pad_nd_image(d[:, s], patch_size, pad_border_mode, pad_kwargs, True,
                                                None)
                    data_shape = data.shape  # still c, x, y
                    steps = compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
                    num_tiles = len(steps[0]) * len(steps[1])
                    add_for_nb_of_preds = np.ones(data.shape[1:], dtype=np.float32)
                    aggregated_results = np.zeros([num_classes] + list(data.shape[1:]),
                                                dtype=np.float32)
                    aggregated_nb_of_predictions = np.zeros([num_classes] + list(data.shape[1:]),
                                                            dtype=np.float32)

                    for step_x in steps[0]:
                        lb_x = step_x
                        ub_x = step_x + patch_size[0]
                        for step_y in steps[1]:
                            lb_y = step_y
                            ub_y = step_y + patch_size[1]

                            x = torch.from_numpy(data[None, :, lb_x:ub_x, lb_y:ub_y])
                            result_torch = torch.zeros([x.shape[0], num_classes] + list(x.shape[2:]),
                                                    dtype=torch.float)
                            x_1 = torch.flip(x, (3, ))
                            x_2 = torch.flip(x, (2, ))
                            x_3 = torch.flip(x, (3, 2))
                            mirrors_input = torch.cat([x,x_1,x_2,x_3], dim=0)
                            pre_data = mirrors_input.numpy()
                            pre_data = pre_data.transpose([0, 2, 3, 1])
                            for k in range(len([pre_data])):
                                inputs[k].from_numpy([pre_data][k])
                                inputs[k].to(dev)
                            status =context.enqueue(inputs, outputs, queue)
                            queue.sync()
                            if not status.ok():
                                print("status is not ok!")
                            result = []
                            for out in outputs:
                                if isinstance(out, list):
                                    for t in out:
                                        result.append(t.asnumpy())
                                else:
                                    result.append(out.asnumpy())
                            pred_np = result[0].transpose([0,3,1,2])
                            for m in range(num_results):
                                y = torch.from_numpy(pred_np[m:m+1,:,:,:])
                                pred = inference_apply_nonlin(y)
                                if m == 0:
                                    result_torch += 1 / num_results * pred
                                elif m == 1:
                                    result_torch += 1 / num_results * torch.flip(pred, (3, ))
                                elif m == 2:
                                    result_torch += 1 / num_results * torch.flip(pred, (3, ))
                                else:
                                    result_torch += 1 / num_results * torch.flip(pred, (3, 2))
                            # _internal_predict_2D_2Dconv_tiled
                            predicted_patch = result_torch[0].detach().cpu().numpy()
                            aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                            aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds
                    # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
                    slicer = tuple([
                        slice(0, aggregated_results.shape[i])
                        for i in range(len(aggregated_results.shape) - (len(slicer) - 1))
                    ] + slicer[1:])
                    aggregated_results = aggregated_results[slicer]
                    aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

                    # computing the class_probabilities by dividing the aggregated result with result_numsamples
                    class_probabilities = aggregated_results / aggregated_nb_of_predictions

                    if regions_class_order is None:
                        pred_seg = class_probabilities.argmax(0)
                    else:
                        class_probabilities_here = class_probabilities
                        pred_seg = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
                        for i, c in enumerate(regions_class_order):
                            pred_seg[class_probabilities_here[i] > 0.5] = c

                    predicted_segmentation.append(pred_seg[None])
                    softmax_pred.append(class_probabilities[None])
                predicted_segmentation = np.vstack(predicted_segmentation)
                softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
                softmaxes.append(softmax_pred)
                softmax = sum(softmaxes)
                if transpose_forward is not None:
                    softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])
                npz_file = None
                region_class_order = None
                bytes_per_voxel = 4
                save_segmentation_nifti_from_softmax(softmax, output_filename, dct, interpolation_order, \
                    region_class_order, None, None, npz_file, None, force_separate_z, interpolation_order_z)

    evaluate_folder(folder_with_gts=args.ref_folder,
                    folder_with_predictions=args.output_folder,
                    labels=[1])

    with open(args.output_folder + '/summary.json', 'r') as f:
        import json
        jstr = json.load(f)
        print('mean acc:', jstr['results']['mean'])
        print('see', args.output_folder + '/summary.json', 'for detail.')

