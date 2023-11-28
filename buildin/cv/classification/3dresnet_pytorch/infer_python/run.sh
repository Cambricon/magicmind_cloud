#!/bin/bash
set -e
set -x

magicmind_model=${1}
batch_size=${2}
json_name="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model}).json"
infer_res_dir="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_infer_res"
if [ ! -d ${infer_res_dir} ];
then
  mkdir -p ${infer_res_dir}
fi

cd ${PROJ_ROOT_PATH}/export_model/3D-ResNets-PyTorch
if [ ! -f inference_mlu.py ];then
    ln -s ${PROJ_ROOT_PATH}/infer_python/inference.py ./inference_mlu.py
fi

python main.py --root_path ./data \
               --result_path ${infer_res_dir} \
               --inference_batch_size ${batch_size} \
               --magicmind_model ${magicmind_model} \
               --video_path kinetics_videos/jpg \
               --annotation_path kinetics.json \
               --dataset kinetics \
               --resume_path weights/r3d50_K_200ep.pth \
               --model_depth 50 \
               --n_classes 700 \
               --n_threads 4 \
               --no_train \
               --no_val \
               --inference \
               --output_topk 5 \
               --no_cuda \
               --use_mlu

python -m util_scripts.eval_accuracy ./data/kinetics.json ${infer_res_dir}/val.json --subset validation -k 1 --ignore
python -m util_scripts.eval_accuracy ./data/kinetics.json ${infer_res_dir}/val.json --subset validation -k 5 --ignore
