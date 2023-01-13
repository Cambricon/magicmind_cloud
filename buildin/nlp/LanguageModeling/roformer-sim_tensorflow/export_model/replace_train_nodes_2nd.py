import tensorflow.compat.v1 as tf
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops
import six as _six
from tensorflow.python.framework import graph_io

input_model = "../data/models/roformer_const.pb"
output_model_path = "../data/models/"
output_model_name = "sim_finish.pb"

def _to_bytes(s):
  """Encode s if it is a sequence of chars."""
  if isinstance(s, _six.text_type):
    return s.encode("utf-8", errors="surrogateescape")
  return s

with tf.gfile.FastGFile(input_model, 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

graph = tf.Graph()
with graph.as_default():
  tf.import_graph_def(graph_def, name="")
  meta_graph_def = saver.export_meta_graph(
    graph_def=graph.as_graph_def(add_shapes=True), graph=graph)

  grappler_session_config = tf.ConfigProto()
  rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
  rewriter_config_with_trt.optimizers.extend(
    ["MLUControlOptimizer"])
  rewriter_config_with_trt.meta_optimizer_iterations = (
       rewriter_config_pb2.RewriterConfig.ONE)
  #rewriter_config_with_trt.custom_optimizers.add().name = "pruning"

  grappler_session_config.graph_options.rewrite_options.CopyFrom(rewriter_config_with_trt)

  collection_def = meta_graph_def.collection_def["train_op"]
  blacklist = collection_def.node_list.value
  for i in ["Pooler-Dense/BiasAdd"]:
    if isinstance(i, ops.Tensor):
      blacklist.append(_to_bytes(i.name))
    else:
      blacklist.append(_to_bytes(i))

output_graph =  tf_optimizer.OptimizeGraph(
           grappler_session_config, meta_graph_def, graph_id=b"tf_graph")

graph_io.write_graph(output_graph, output_model_path,name = output_model_name,as_text = False)

print("frozen pb finish ")