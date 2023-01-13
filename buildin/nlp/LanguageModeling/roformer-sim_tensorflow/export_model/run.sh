#!/bin/bash
set -e
set -x

if [ ! -d $MODEL_PATH ];
then
  mkdir -p $MODEL_PATH
fi 
cd $MODEL_PATH
if [ ! -f "sim_finish.pb" ];
then
    cd $PROJ_ROOT_PATH/export_model
    bash get_models.sh
    python retrieval.py
    python replace_train_nodes_1st.py
    python replace_train_nodes_2nd.py
else
    echo "pb has been exported !"
fi
