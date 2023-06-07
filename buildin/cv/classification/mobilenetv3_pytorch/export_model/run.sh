#bin/bash
set -e
set -x
bash get_datasets_and_models.sh

cd ${PROJ_ROOT_PATH}/export_model
if [ ! -f ${MODEL_PATH}/mobilenet-v3_small.torchscript.pt ];then
  echo "converting pth to pt"
  python covert.py ${PROJ_ROOT_PATH}
  echo "convert pth to pt success!"
fi 

