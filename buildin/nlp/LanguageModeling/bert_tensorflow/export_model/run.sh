#!/bin/bash
set -e
set -x

# 下载bert-base-cased初始权重
if [ ! -d $PROJ_ROOT_PATH/data/models ];
then
    mkdir $PROJ_ROOT_PATH/data/models
fi
bash ./get_datasets_and_models.sh
