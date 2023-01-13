#!/bin/bash
if grep -q "mm_infer" $PROJ_ROOT_PATH/export_model/DB/eval.py;
then
  echo "infer.patch already be used"
else
  patch -p0 $PROJ_ROOT_PATH/export_model/DB/eval.py < $PROJ_ROOT_PATH/infer_python/infer.patch
fi

PRECISION=$1 
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir $PROJ_ROOT_PATH/data/output
fi
cd $PROJ_ROOT_PATH/export_model/DB
python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml  \
               --polygon \
               --box_thresh 0.7 \
               --model_path ${MODEL_PATH}/dbnet_pt_model_${PRECISION}_true \
               --result_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_true_log_eval
