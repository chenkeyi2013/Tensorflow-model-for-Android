import tensorflow as tf

def load_pb(model_file,sess):
    #MODEL_FILE:pb文件路径
    #sess:会话
    with tf.gfile.FastGFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")



def load_ckpt(model_file,sess):
    # MODEL_FILE:CKPT路径,like '/home/chenkeyi/Workplace/google/model/mobilenet_v1_1.0_224.ckpt'
    # sess:将模型导入至sess
    saver = tf.train.import_meta_graph(model_file + '.meta', clear_devices=True)
    saver.restore(sess, model_file)
