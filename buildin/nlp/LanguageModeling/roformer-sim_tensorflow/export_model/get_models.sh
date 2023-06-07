set -e 
set -x

if [ ! -d $MODEL_PATH ];
then
  mkdir -p $MODEL_PATH
fi 
cd $MODEL_PATH

TF_MODEL_PATH=chinese_roformer-sim-char-ft_L-6_H-384_A-6
if [ ! -d $TF_MODEL_PATH ];
then
    wget -c https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip
    unzip -o chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip  
fi
