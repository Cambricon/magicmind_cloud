#!/bin/bash
set -e
set -x

pb_path=${MODEL_PATH}/frozen_graph.pb
# 下载bert-base-cased初始权重
bash get_datasets_and_models.sh
# 转成PB
if [ -f ${pb_path} ]; 
then
  echo "frozen_graph.pb file already exists."
else
  # 拉取bert仓库
  if [ ! -d 'bert' ];then
    git clone https://github.com/google-research/bert.git
    # 加入frozen pb的代码
    cd bert && git apply ../run_squad.py.patch
  else
    cd bert
  fi

  # 执行脚本生成pb, 需要在tf1的环境下转换
  pip install numpy==1.19.5 -i https://pypi.douban.com/simple
  pip install 'tensorflow-cpu==1.15.0' -i https://pypi.douban.com/simple
  python run_squad.py --vocab_file=${MODEL_PATH}/squad_model/vocab.txt \
                      --bert_config_file=${MODEL_PATH}/squad_model/bert_config.json \
                      --init_checkpoint=${MODEL_PATH}/squad_model/model.ckpt-5474 \
                      --do_train=False --do_predict=True \
                      --predict_file=${SQUAD_DATASETS_PATH}/dev-v1.1.json \
                      --max_seq_length=384 \
                      --output_dir=${MODEL_PATH}
fi