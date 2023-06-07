#!/bin/bash
set -e
set -x

if [ ! -d ${MODEL_PATH} ];
then
  mkdir -p ${MODEL_PATH}
fi 
cd ${MODEL_PATH}
if [ ! -f "sim_finish.pb" ];
then
    bash get_models.sh
    python retrieval.py
    python replace_train_nodes_1st.py
    python replace_train_nodes_2nd.py
    rm ${MODEL_PATH}/roformer*.pb
else
    echo "pb has been exported !"
fi
