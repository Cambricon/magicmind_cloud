#!/bin/bash
set -e
set -x
cd $PROJ_ROOT_PATH/export_model
bash run.sh 1
for precision in force_float32 force_float16 qint8_mixed_float16
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision true 1 0.001 0.65 1000
    for batch in 1
    do
        cd $PROJ_ROOT_PATH/infer_cpp
        bash run.sh $precision true $batch -1
        cd $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface/widerface_evaluate
        python3 setup.py build_ext --inplace
        python3 evaluation.py -p $PROJ_ROOT_PATH/data/output/infer_cpp_output_${precision}_true_${batch}/pred_txts \
                                -g $PROJ_ROOT_PATH/export_model/Pytorch_Retinaface/widerface_evaluate/ground_truth
    done
done