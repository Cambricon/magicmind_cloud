#! -*- coding: utf-8 -*-
# 测试有监督版RoFormer-Sim-FT的相似度效果

from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import tensorflow as tf

maxlen = 64
# bert配置
config_path = '../data/models/chinese_roformer-sim-char-ft_L-6_H-384_A-6/bert_config.json'
checkpoint_path = '../data/models/chinese_roformer-sim-char-ft_L-6_H-384_A-6/bert_model.ckpt'
dict_path = '../data/models/chinese_roformer-sim-char-ft_L-6_H-384_A-6/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])

out_nodes = []
for i in range(len(encoder.outputs)):
    out_nodes.append('output_' + str(i + 1))
    tf.identity(encoder.output[i],'output_' + str(i + 1))
sess = K.get_session()
# sess = tf.compat.v1.Session()
from tensorflow.python.framework import graph_util
init_graph = sess.graph.as_graph_def()
main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)

tf.compat.v1.graph_util.remove_training_nodes(
    main_graph, protected_nodes=None
)
with tf.gfile.GFile('../data/models/roformer.pb', 'wb') as fw:
    fw.write(main_graph.SerializeToString())

print("keras ckpt convert to tensorflow pb SUCCESSED!")