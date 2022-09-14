export NEUWARE_HOME=/usr/local/neuware
export PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
export MAGICMIND_CLOUD="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
export DATASETS_PATH=/nfsdata/modelzoo/datasets/imagenet1000
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/cv/utils 
export MM_RUN_PATH=$NEUWARE_HOME/bin
export MODEL_NAME=resnext50_caffe
