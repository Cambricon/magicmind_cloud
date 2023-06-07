#!/bin/bash
set -e
set -x


### install dependencies
cd ${PROJ_ROOT_PATH}/export_model
if [ -x wenet ]; then
    echo "WeNet Official repo already exists."
else
    echo "get WeNet..."
    git clone -b v2.0.0 https://github.com/wenet-e2e/wenet.git
fi

# patch-wenet
if grep -q "opset_version=11" ${PROJ_ROOT_PATH}/export_model/wenet/wenet/bin/export_onnx_gpu.py;
then 
  echo "Patch already applied"
else
  echo "modifying the WeNet..."
  cd ${PROJ_ROOT_PATH}/export_model/wenet
  git apply  ${PROJ_ROOT_PATH}/export_model/patchs/mlu.patch
fi

### get pretrained model
if [ ! -d ${MODEL_PATH} ];
then
    mkdir -p ${MODEL_PATH}
fi

cd ${MODEL_PATH}
if [ -x "20211025_conformer_exp" ]
then 
    echo "Pretrained models exists."
else
    wget http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/wenetspeech/20211025_conformer_exp.tar.gz
    echo "Download pretrain models 20211025_conformer_exp.tar.gz."
    tar zxvf 20211025_conformer_exp.tar.gz
    echo "Extract model from 20211025_conformer_exp.tar.gz. "
    rm 20211025_conformer_exp.tar.gz
    echo "Delete tmp model file."
fi

### prepare data
if [ ! -d ${AISHELL_DATASETS_PATH} ];
then
    mkdir -p ${AISHELL_DATASETS_PATH}
fi

### 1. download datas
s0_path=${PROJ_ROOT_PATH}/export_model/wenet/examples/aishell/s0
if [ -x ${AISHELL_DATASETS_PATH}/data_aishell ]
then 
    echo "Dataset aishell already exists"
else
    # make sure of using absolute path. DO-NOT-USE relatvie path!
    data_url=www.openslr.org/resources/33
    ${s0_path}/local/download_and_untar.sh ${AISHELL_DATASETS_PATH} ${data_url} data_aishell
    echo "download and untar aishell dataset ..."
fi

### 2. process data
if [ -x ${s0_path}/data ]
then 
    echo "data list already exists"
else
    cd ${s0_path}
    ${s0_path}/local/aishell_data_prep.sh ${AISHELL_DATASETS_PATH}/data_aishell/wav/ ${AISHELL_DATASETS_PATH}/data_aishell/transcript/
    ### 3. get test data.list
    ${s0_path}/tools/make_raw_list.py ${s0_path}/data/local/test/wav.scp ${s0_path}/data/local/test/text ${s0_path}/data/local/test/data.list
    echo "Test set data.list  preparation succeeded"
fi



cd ${PROJ_ROOT_PATH}/export_model
### install ctc_decoder
if [ -x ctc_decoder ];
then
    echo "ctc_decoder already exists"
else
    git clone https://github.com/Slyne/ctc_decoder.git
    cd ctc_decoder
    git checkout 68f8ba
    apt-get update
    apt update
    apt-get install swig -y
    apt-get install python3-dev -y
    
    if [ $? -ne 0 ]; then
        echo 'apt update failed, check your network connection please!!!'
        exit 1
    fi
    cd swig && bash setup.sh
fi

echo "export onnx models for asr encoder & decoder."

### pytorch models convert to onnx models
if [ -f "${MODEL_PATH}/encoder.onnx" -a -f "${MODEL_PATH}/encoder.onnx" ];
then
    echo "onnx already exists"
else
    cd ${PROJ_ROOT_PATH}/export_model/wenet
    python ./wenet/bin/export_onnx_gpu.py --config ${MODEL_PATH}/20211025_conformer_exp/train.yaml \
                                       --checkpoint ${MODEL_PATH}/20211025_conformer_exp/final.pt  \
                                       --beam 4  \
                                       --output_onnx_dir ${MODEL_PATH}/ \
                                       --cmvn_file ${MODEL_PATH}/20211025_conformer_exp/global_cmvn
fi
