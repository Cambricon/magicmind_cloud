#bin/bash
set -e
set -x


# 2.下载clip 实现源码
cd $PROJ_ROOT_PATH/export_model
if [ -d "CLIP" ];
then
  echo "clip already exists."
else
  echo "git clone clip..."
  git clone https://github.com/openai/CLIP.git
  cd $PROJ_ROOT_PATH/export_model/CLIP
  git checkout d50d76daa670286dd6cacf3bcd80b5e4823fc8e1
  python setup.py install 
  
fi
cd $PROJ_ROOT_PATH/export_model
echo "DownLoading and Converting Models..."
# download datasets and convert torch model to pt model
if [ ! -d $PROJ_ROOT_PATH/data/models ];then
    mkdir -p $PROJ_ROOT_PATH/data/models
fi
python download_and_convert_model.py $PROJ_ROOT_PATH
python download_datasets.py $CIFAR100_DATASETS_PATH

