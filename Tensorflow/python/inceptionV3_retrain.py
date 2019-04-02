#%%
import glob
import numpy as np
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import data_op as cky
from tensorflow.python.framework import graph_util
#加载通过TensorFlow-Slim定义好的inception_v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

#%%

#处理好之后的tfrecords文件
INPUT_DATA = ["D:\\WorkPlace\\google_workplace\\v1.3\\record_v1.3\\train.tfrecords","D:\\WorkPlace\\google_workplace\\v1.3\\record_v1.3\\validation.tfrecords","D:\\WorkPlace\\google_workplace\\v1.3\\record_v1.3\\test.tfrecords"]

#数据集数量
n_train_set = 2414
n_validation_set = 293
n_test_set = 293

#保存训练好的模型路径。将使用新数据训练得到的完整模型保存下来，如果有余力，在全连接层的训练完成之后再训练全部网络层
TRAIN_FILE = 'D:\\WorkPlace\\google_workplace\\v1.3\\ckpt\\model.ckpt'
#谷歌提供的模型
CKPT_FILE = 'D:\\WorkPlace\\google_workplace\\v1.3\\inception_v3.ckpt'
#pb文件保存路径
PB_FILE = 'D:\\WorkPlace\\google_workplace\\v1.3\\test.pb'


#训练中的超参数
LEARNING_RATE = 0.0001
STEPS = 500
BATCH = 6
N_CLASSES = 3

#%%
#不需要从下载的模型中加载的参数，这里就是最后的全连接层，因为新的问题中要重新训练这一层的参数，这里给出的是参数前缀
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'


# 获取所有需要从谷歌训练好的模型中加载的参数
def get_tuned_variables():
    # exclusions--列表:['InceptionV3/Logits', 'InceptionV3/AuxLogits']
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []
    # 枚举Inception_V3模型中的所有参数，然后判断是否需要从加载列表中移除
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore
#%%
#定义Inception-v3的输入，images为输入图片，labels为图片标签

#images维度为Inception-v3所要求的299,299,3
images = tf.placeholder(tf.float32,[None,299,299,3],name='input_images')
labels = tf.placeholder(tf.int64,[None],name = 'labels')
#%%
# g1 当前运算,保存时候用的图
g1 = tf.get_default_graph()
# g2 无关Tensor的垃圾桶
g2 = tf.Graph()
#%%
#获取数据集
train_set = cky.get_set(INPUT_DATA[0],n_train_set)
validation_set = cky.get_set(INPUT_DATA[1],n_validation_set)
test_set = cky.get_set(INPUT_DATA[2],n_test_set)
#读取数据  train数据在训练过程中抓取Batch数量
with g2.as_default():
    validation_images,validation_labels = cky.get_data(validation_set,n_validation_set)
    test_images,test_labels = cky.get_data(test_set,n_test_set)

#print(tf.get_default_graph() is g1)
#print(test_labels)
#%%

#定义Inception3-v3模型
with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, _ = inception_v3.inception_v3(inputs = images, num_classes=N_CLASSES)

#%%

#定义交叉熵损失函数，注意在模型定义的时候已经将正则化损失加入了损失集合中
tf.losses.softmax_cross_entropy(
tf.one_hot(labels,N_CLASSES),logits,weights=1.0)
#%%

#定义训练过程
train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

#%%

#计算正确率

with tf.name_scope('evaluation'):
    pre = tf.argmax(logits,1,name = 'output')
    correct_prediction = tf.equal(pre,labels)
    evalution_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#%%

#定义加载模型的函数
load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE,get_tuned_variables(),ignore_missing_vars=True)

#%%
with g1.as_default():
    saver = tf.train.Saver()

    sess = tf.Session(graph=g1)

    # 初始化没有加载进来的变量，注意这个过程一定要在模型加载之前，否则初始化过程会将已经加载的变量重新赋值
    init = tf.global_variables_initializer()
    sess.run(init)
    # 加载已经训练好的模型
    print('Loading tuned variables from %s' % CKPT_FILE)
    load_fn(sess)

    for i in range(STEPS):
        with g2.as_default():
            train_images,train_labels = cky.get_data(train_set,BATCH)

        sess.run(train_step,feed_dict={
            images: train_images,
            labels: train_labels
        })

        if i % 50 == 0 or i + 1 == STEPS:
            saver.save(sess,TRAIN_FILE,global_step = i)

            upper = int(n_validation_set / 3.0)
            downer = 0
            whole_accuracy = 0.0
            for j in range(3):
                accuracy = sess.run(evalution_step,feed_dict={
                    images: validation_images[downer:upper],
                    labels: validation_labels[downer:upper]
                })
                whole_accuracy += accuracy

                downer = upper
                upper += int(n_validation_set / 3.0)
            print('Step %d: Validation accuracy = %.lf%%' % (i,(whole_accuracy / 3.0) * 100))

    upper = int(n_test_set / 3.0)
    downer = 0
    whole_accuracy = 0.0
    for j in range(3):
        accuracy = sess.run(evalution_step,feed_dict={
            images: test_images[downer:upper],
            labels: test_labels[downer:upper]
        })
        whole_accuracy += accuracy

        downer = upper
        upper += int(n_test_set / 3.0)
    print('----------------------------------------')
    print('After Step %d,the test accuracy = %.lf%%' % (STEPS,(whole_accuracy / 3.0) * 100))
    print('----------------------------------------')







#%%
output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=['InceptionV3/Logits/SpatialSqueeze'])
with tf.gfile.GFile(PB_FILE, mode='wb') as f:#’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    f.write(output_graph_def.SerializeToString())