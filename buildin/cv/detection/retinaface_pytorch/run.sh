#!/bin/bash
set -e
set -x

precision=qint8_mixed_float16
dynamic_shape=false
batch_size=1
image_num=-1

magicmind_model=${MODEL_PATH}/retinaface_pytorch_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_model="${magicmind_model}_${batch_size}"
fi

# 0. export model
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh ${batch_size}

# 1. gen_model
if [ ! -f ${magicmind_model} ];then
    cd ${PROJ_ROOT_PATH}/gen_model
    bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}    
else
    echo "MagicMind model: ${magicmind_model} already exists!"
fi

# 2. build infer_cpp and eval
cd ${PROJ_ROOT_PATH}/infer_cpp
bash run.sh ${magicmind_model} ${batch_size} 
