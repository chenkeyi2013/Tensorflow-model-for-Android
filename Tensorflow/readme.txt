-----------------------------------------------------------------------------------------------------------------------------------

This Folder is make up of :

1.mobilenet_model

  the tuned mobilenet model,download of Tensorflow Lite website.

2.InceptionV3_model
  
  the tuned InceptionV3_model,download of Tensorflow website.

3.python
  the python code,include of how to feed your own picture to model,how to retrain the tuned model,and some problem you maybe encounter.

4.CKPT

  the CKPT file after retrain.

-----------------------------------------------------------------------------------------------------------------------------------

1:
  be careful about your memory.you should free the tensor after load tfrecord when your data is too large.

2:
  Tensorflow is a framework that the tensor and calculate are separated.
  So maybe you will get some strange bug when you write your own code,good luck :)
  
3.
  use tf.get_tensor_by_name to get the tensor you wanted.
  use tf.get_collection(tf.GraphKeys().trainable_variables) to check what tensor are not frozen.

4.
  RMSPropOptimizer define some dropout variables in graph,and you need custom the operation if you want to convert your model
  to tf.lite.
  so,remove it before you get your frozen graph.
