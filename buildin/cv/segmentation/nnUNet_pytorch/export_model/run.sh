#!/bin/bash
set -e
set -x
PARAMETER_ID=$1
if [ ! -d $PROJ_ROOT_PATH/data ];
then
    mkdir $PROJ_ROOT_PATH/data
fi

if [ ! -d $PROJ_ROOT_PATH/data/models ];
then
    mkdir $PROJ_ROOT_PATH/data/models
fi

if [ ! -d $PROJ_ROOT_PATH/data/models/saved_pts ];
then 
    mkdir $PROJ_ROOT_PATH/data/models/saved_pts
fi

if [ -f "$PROJ_ROOT_PATH/data/models/saved_pts/2dunet_${PARAMETER_ID}.pt" ];
then echo "nnUNet pt already exists."
else 
    # 1.下载数据集和模型
    bash get_datasets_and_models.sh

    # 2.下载并安装nnUNet
    cd $PROJ_ROOT_PATH/export_model
    if [ -d "nnUNet" ];
    then
        echo "nnUNet already exists."
    else
        echo "git clone nnUNet..."
        git clone https://github.com/MIC-DKFZ/nnUNet.git
    fi
    cd nnUNet
    git reset --hard  b16142ac0d15e4098d9b6c9a2b828b8dc4957c2f
    pip install -e .

    # 3.patch-nnUNet
    if grep -q "torch.jit.trace" $PROJ_ROOT_PATH/export_model/nnUNet/nnunet/network_architecture/neural_network.py;
    then 
        echo "modifying the nnUNet has been already done"
    else
        echo "modifying the nnUNet..."
        patch $PROJ_ROOT_PATH/export_model/nnUNet/nnunet/network_architecture/neural_network.py $PROJ_ROOT_PATH/export_model/neural_network.patch
    fi

    # 4. 按照nnUNet格式要求对数据集进行预处理
    if [ ! -d $nnUNet_raw_data_base ];
    then
	    cp -r $NNUNET_nnUNet_raw_data_base/../nnUNet_raw_data_base $DATASETS_PATH
    fi

    nnUNet_convert_decathlon_task -i $nnUNet_raw_data_base/Task02_Heart
    nnUNet_plan_and_preprocess -t 2 --verify_dataset_integrity

    # 5.trace model
    cd $PROJ_ROOT_PATH/export_model
    # FIXME: An error will be reported when executing the following command
    # The specific error is as follows
    # FileNotFoundError: [Errno 2] No such file or directory:
    # magicmind_cloud/buildin/cv/segmentation/nnUNet_pytorch/data/models/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl
    python export.py -o $PROJ_ROOT_PATH/data/models/saved_pts \
                     -i $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
                     -m $MODEL_PATH/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1 \
                     --parameter_id ${PARAMETER_ID}
fi
