#!/bin/bash
set -e
set -x
if [ ! -d $PROJ_ROOT_PATH/data ];
then 
    mkdir $PROJ_ROOT_PATH/data
fi
bash get_datasets_and_models.sh
