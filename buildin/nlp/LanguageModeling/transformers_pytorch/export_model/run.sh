#bin/bash
set -e
set -x
echo "DownLoading and Converting Models..."
# download datasets and convert torch model to pt model
if [ ! -d $PROJ_ROOT_PATH/data/models ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi

if [ ! -f $PROJ_ROOT_PATH/export_model/glue.py ];then
    wget -c https://raw.githubusercontent.com/huggingface/datasets/main/metrics/glue/glue.py -O $PROJ_ROOT_PATH/export_model/glue.py
fi

python download_and_convert_model.py $PROJ_ROOT_PATH

if [ ! -d $PROJ_ROOT_PATH/data/glue_data ];then
  cp -r $GLUE_DATASETS_PATH $PROJ_ROOT_PATH/data
fi

python download_datasets.py $PROJ_ROOT_PATH/data/glue_data


