#bin/bash
set -e
set -x

FILE="slot_test_data_full"

if [ ! -d ${CRITEO_DATASETS_PATH} ];then
   mkdir -p ${CRITEO_DATASETS_PATH}
fi 
cd ${CRITEO_DATASETS_PATH}

if [ ! -d ${FILE} ];then 
   echo "Downloading slot_test_data_full.tar.gz"
   wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_test_data_full.tar.gz
   tar xzvf slot_test_data_full.tar.gz
fi

if [ ! -d ${MODEL_PATH} ];then
   mkdir -p ${MODEL_PATH}
fi 

cd ${MODEL_PATH}
#Users should prepare the xdeepfm dygraph model in advance
xDeepFM_paddle_dynamic_model="${MODEL_PATH}/dygraph_model_xdeepfm_all"
if [ ! -d ${xDeepFM_paddle_dynamic_model} ];then
   echo "PaddleRec xDeepFM Dygraph Model is not found, please prepare it!"
   exit 1
fi

xDeepFM_paddle_static_model="${MODEL_PATH}/static_model_xdeepfm_all"
xDeepFM_onnx_model="${MODEL_PATH}/xdeepfm.onnx"

if [ ! -f ${xDeepFM_onnx_model} ];then
   
   if [ ! -d ${xDeepFM_paddle_static_model} ];then
      if [ ! -d "PaddleRec" ];then
         git clone https://github.com/PaddlePaddle/PaddleRec.git
      fi
      cd PaddleRec/models/rank/xdeepfm
      git checkout cd7fdf17fc1afd33f1e505f0359221bcb2928ab0
      # dygraph model to static model
      python ../../../tools/to_static.py -m config_bigdata.yaml \
          -o runner.use_gpu=False runner.model_init_path=${xDeepFM_paddle_dynamic_model} \
          runner.model_save_path=${xDeepFM_paddle_static_model}
   fi

   paddle2onnx --model_dir ${xDeepFM_paddle_static_model}/0 \
            --model_filename tostatic.pdmodel \
            --params_filename tostatic.pdiparams \
            --save_file ${xDeepFM_onnx_model} \
            --opset_version 11 \
            --enable_dev_version True
fi
