#bin/bash
set -e
set -x
## get model and datasets
bash get_datasets_and_models.sh

cd $MODEL_PATH

FILE="yolov3_nonms.onnx"
if [ ! -f $FILE ];then
  echo $FILE
  cd $MODEL_PATH/PaddleDetection
  paddle2onnx --model_dir inference_model_nonms/yolov3_darknet53_270e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file yolov3_nonms.onnx
  mv yolov3_nonms.onnx ../
fi

