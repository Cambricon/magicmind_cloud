#!/bin/bash
PRECISION=$1

if [ ! -d "$PROJ_ROOT_PATH/data/output_${PRECISION}" ];
then
  mkdir "$PROJ_ROOT_PATH/data/output_${PRECISION}"
  echo "mkdir sucessed!!!"
else
  echo "output dir exits!!! no need to mkdir again!!!"
fi
MAGICMIND_MODEL=$MODEL_PATH/3dresnet_${PRECISION}_1.mm
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/
if [ ! -d "$OUTPUT_DIR" ];
then
  mkdir "$OUTPUT_DIR"
fi
echo "infer Magicmind model..."
cd $PROJ_ROOT_PATH/export_model/3D-ResNets-PyTorch
ln -s $PROJ_ROOT_PATH/infer_python/inference.py ./inference_mlu.py
python main.py --root_path ./data \
               --video_path kinetics_videos/jpg \
               --annotation_path kinetics.json \
	       --result_path ../../../data/output_${PRECISION} \
	       --dataset kinetics \
	       --resume_path weights/r3d50_K_200ep.pth \
	       --model_depth 50 \
	       --n_classes 700 \
	       --n_threads 4 \
	       --no_train \
	       --no_val \
	       --inference \
	       --output_topk 5 \
	       --inference_batch_size 1 \
	       --no_cuda \
	       --magicmind_model $MAGICMIND_MODEL \
	       --use_mlu
python -m util_scripts.eval_accuracy ./data/kinetics.json ../../data/output_${PRECISION}/val.json --subset validation -k 1 --ignore
python -m util_scripts.eval_accuracy ./data/kinetics.json ../../data/output_${PRECISION}/val.json --subset validation -k 5 --ignore
