#bin/bash
set -e
set -x
echo "DownLoading and Converting Models..."
# download datasets and convert torch model to pt model
if [ ! -d $PROJ_ROOT_PATH/data/models ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
python download_and_convert_model.py $PROJ_ROOT_PATH
python download_datasets.py $DATASETS_PATH

