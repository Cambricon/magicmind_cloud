### �ڿ�ʼ���б��ֿ�ǰ�ȼ������·����
export NEUWARE_HOME=/usr/local/neuware/
##MM_RUN·��
export MM_RUN_PATH=$NEUWARE_HOME/bin
#��sample����·��
export MAGICMIND_CLOUD=${PWD}/../../../../../magicmind_cloud
export PROJ_ROOT_PATH=$MAGICMIND_CLOUD/buildin/cv/classification/googlenet_bn_caffe
#���ݼ�·��
export DATASETS_PATH=/nfsdata/modelzoo/datasets/ILSVRC2012
#ģ��·��
export MODEL_PATH=$PROJ_ROOT_PATH/data/models
#cv������ͨ���ļ�·��
export UTILS_PATH=$MAGICMIND_CLOUD/buildin/cv/utils
echo "check below paths before run this sample!!!"
echo "DATASETS_PATH now is $DATASETS_PATH, please replace it to path where you want to save datasets"
echo "NEUWARE_HOME now is $NEUWARE_HOME"
echo "MM_RUN_PATH now is $MM_RUN_PATH"
echo "MAGICMIND_CLOUD is $MAGICMIND_CLOUD"
echo "PROJ_ROOT_PATH is $PROJ_ROOT_PATH"
echo "MODEL_PATH is $MODEL_PATH"
echo "UTILS_PATH is $UTILS_PATH"
