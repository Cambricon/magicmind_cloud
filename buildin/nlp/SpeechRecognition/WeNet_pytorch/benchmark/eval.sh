#!/bin/bash
set -e
set -x



if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi

cd $PROJ_ROOT_PATH/export_model
bash run.sh


for precision in force_float32
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh ${precision}
    cd  ${PROJ_ROOT_PATH}/infer_python
    bash run.sh ${precision}
done

