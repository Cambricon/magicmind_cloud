import tensorflow.compat.v1 as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util

input_model = "../data/models/roformer.pb"
output_model_path = "../data/models/"
output_model_name = "roformer_const.pb"

def create_node(op, name, inputs):
  new_node = node_def_pb2.NodeDef()
  new_node.op = op
  new_node.name = name
  for input_name in inputs:
    new_node.input.extend([input_name])
  return new_node

def create_constant_node(name, value, dtype, shape=None):
  node = create_node("Const", name, [])
  set_attr_dtype(node, "dtype", dtype)
  set_attr_tensor(node, "value", value, dtype, shape)
  return node

def set_attr_dtype(node, key, value):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(type=value.as_datatype_enum))
  except KeyError:
    pass

def set_attr_tensor(node, key, value, dtype, shape=None):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            value, dtype=dtype, shape=shape)))
  except KeyError:
    pass

with tf.Graph().as_default() as graph:
  in_graph_def = tf.GraphDef()
  out_graph_def = tf.GraphDef()
  with tf.gfile.GFile(input_model, 'rb') as fid:
    serialized_graph = fid.read()
    in_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(in_graph_def, name='')
    for node in in_graph_def.node:
      if node.name in ["keras_learning_phase"]:
        new_const = create_constant_node(node.name, False, tf.bool, shape=[])
        out_graph_def.node.extend([new_const])
      else:
        out_graph_def.node.extend([node])
    if in_graph_def.library:
      out_graph_def.library.CopyFrom(in_graph_def.library)     
    
    tf.train.write_graph(out_graph_def, output_model_path , output_model_name, False)
    
print("stage 1 finish ")
