import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from nnunet.preprocessing.preprocessing import PreprocessorFor2D
from copy import deepcopy


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids

def compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
    import numpy as np
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps

def preprocess(args):
    info = load_pickle(join(args.model_path + "/fold_0/", "%s.model.pkl" % "model_final_checkpoint"))
    maybe_mkdir_p(args.output_folder)
    shutil.copy(join(args.model_path, 'plans.pkl'), args.output_folder)

    assert isfile(join(args.model_path,
                    "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(args.model_path, "plans.pkl"))['num_modalities']

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(args.data_folder, expected_num_modalities)

    output_files = [join(args.output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(args.data_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[
        join(args.data_folder, i) for i in all_files
        if i[:len(j)].startswith(j) and len(i) == (len(j) + 12)
    ] for j in case_ids]

    plans = info['plans']
    normalization_schemes = plans['normalization_schemes']
    use_mask_for_norm = plans['use_mask_for_norm']
    transpose_forward = plans['transpose_forward']
    intensity_properties = plans['dataset_properties']['intensityproperties']
    transpose_backward = plans['transpose_backward']

    init = info['init']
    _, _, _, _, _, stage, _, _, _ = init
    stage_plans = plans['plans_per_stage'][stage]
    batch_size = stage_plans['batch_size']
    net_pool_per_axis = stage_plans['num_pool_per_axis']
    patch_size = np.array(stage_plans['patch_size']).astype(int)
    do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']
    target_spacing = plans['plans_per_stage'][stage]['current_spacing']
    pad_border_mode = 'constant'
    pad_kwargs = {'constant_values': 0}
    step_size = 0.5
    num_classes = 2
    mirror_idx = 4
    softmax_helper = lambda x: F.softmax(x, 1)
    inference_apply_nonlin = softmax_helper
    num_results = 4
    mirror_axes = (0, 1)
    regions_class_order = None

    if 'segmentation_export_params' in plans.keys():
        force_separate_z = plans['segmentation_export_params']['force_separate_z']
        interpolation_order = plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0

    # TODO: extend for 3d
    preprocessor = PreprocessorFor2D(normalization_schemes, use_mask_for_norm, transpose_forward,
                                    intensity_properties)
    return preprocessor, list_of_lists, output_files, target_spacing, patch_size, inference_apply_nonlin, transpose_forward, transpose_backward, interpolation_order, force_separate_z, interpolation_order_z



 
