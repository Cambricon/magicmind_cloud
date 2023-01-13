# 1. prepare data and models
bash get_datasets_and_models.sh

# 2. git clone source code
if [ -d tensorflow_PSENet ];
then
    echo "tensorflow_PSENet already exist."
else
    echo "git clone tensorflow_PSENet"
    git clone https://github.com/liuheng92/tensorflow_PSENet
    cd tensorflow_PSENet
    git reset --hard e2cd908f301b762150aa36893677c1c51c98ff9e
fi


# 3. patch
cd $PROJ_ROOT_PATH/export_model
if grep -q "Sigmoid"  $PROJ_ROOT_PATH/export_model/tensorflow_PSENet/eval.py;
then
    echo "patch already be used"
else
    echo "patching"
    patch -p0 tensorflow_PSENet/eval.py < export.patch
    patch -p0 tensorflow_PSENet/utils/utils_tool.py < queue.patch
fi

# 4. export pb
cd $PROJ_ROOT_PATH/export_model/tensorflow_PSENet
python eval.py  --checkpoint $MODEL_PATH/model/model.ckpt \
                --pb_save_path $MODEL_PATH/psenet.pb