#bin/bash
set -e
set -x
## get model and datasets
bash get_datasets_and_models.sh

cd ${MODEL_PATH}
FILE="xception41.onnx"
if [ ! -f ${FILE} ];then
  echo ${FILE}
  paddle2onnx --model_dir Xception41_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --opset_version 11 --save_file xception41.onnx
fi
