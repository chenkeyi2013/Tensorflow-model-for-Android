#%%
import glob
import numpy as np
import os.path
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import data_op as cky
from tensorflow.python.framework import graph_util
import item.import_pb as import_pb

#%%
# 忽略CPU警告
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU control
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7


# 处理好之后的tfrecords文件
INPUT_DATA = ["/home/chenkeyi/Workplace/google/tfrecords/no_stand/train.tfrecords",
              "/home/chenkeyi/Workplace/google/tfrecords/no_stand/validation.tfrecords",
              "/home/chenkeyi/Workplace/google/tfrecords/no_stand/test.tfrecords",
              "/home/chenkeyi/Workplace/google/tfrecords/cwq/train.tfrecords"]


TENSORBOARD_LOG = '/home/chenkeyi/Workplace/tensorboard/log'

#%%
#数据集数量

n_train_set = 2059
n_validation_set = 463
n_test_set = 478
n_others = 674

#保存训练好的模型路径。将使用新数据训练得到的完整模型保存下来，如果有余力，在全连接层的训练完成之后再训练全部网络层
TRAIN_FILE = '/home/chenkeyi/Workplace/google/ckpt/model.ckpt'
#谷歌提供的模型
MODEL_FILE = '/home/chenkeyi/Workplace/google/model/mobilenet_v1_1.0_224_frozen.pb'

# pb文件保存路径
PB_FILE = '/home/chenkeyi/Workplace/google/test.pb'
# CKPT文件保存路径
CKPT_FILE = '/home/chenkeyi/Workplace/google/model/mobilenet_v1_1.0_224.ckpt'

#训练中的超参数
LEARNING_RATE = 0.0001
STEPS = 1000
BATCH = 6
N_CLASSES = 3


#%%

# g1 当前运算,保存时候用的图
g1 = tf.get_default_graph()
# g2 无关Tensor的垃圾桶
g2 = tf.Graph()


#%%
#获取数据集
with g2.as_default():
    train_set = cky.get_set(INPUT_DATA[0], n_train_set)
    validation_set = cky.get_set(INPUT_DATA[1], n_validation_set)
    test_set = cky.get_set(INPUT_DATA[2], n_test_set)
    others_set = cky.get_set(INPUT_DATA[3], n_others)

    train_images, train_labels = cky.get_data(train_set, n_train_set)
    validation_images, validation_labels = cky.get_data(validation_set, n_validation_set)
    test_images, test_labels = cky.get_data(test_set, n_test_set)
    others_images, others_labels = cky.get_data(others_set, n_others)

#%%
# import model
sess = tf.Session(graph=g1,config=config)
import_pb.load_pb(MODEL_FILE, sess)

#%%
X = g1.get_tensor_by_name('MobilenetV1/Logits/Dropout_1b/Identity:0')

F1 = tf.Variable(tf.random_normal([1, 1, 1024, 1001]))

C1 = tf.nn.conv2d(X, F1, [1, 1, 1, 1], "VALID")

#%%
Bias = tf.Variable(tf.random_normal([1001]))

BiasAdd = tf.nn.bias_add(C1,Bias)

Squeeze = tf.squeeze(BiasAdd,[1,2])
#%%
W1 = tf.Variable(tf.random_normal([1001,512]))
B1 = tf.Variable(tf.zeros([512]))
L1 = (tf.matmul(Squeeze, W1) + B1)

W2 = tf.Variable(tf.random_normal([512,256]))
B2 = tf.Variable(tf.zeros([256]))
L2 = (tf.matmul(L1, W2) + B2)

W3 = tf.Variable(tf.random_normal([256,128]))
B3 = tf.Variable(tf.zeros([128]))
L3 = (tf.matmul(L2, W3) + B3)

W4 = tf.Variable(tf.random_normal([128,64]))
B4 = tf.Variable(tf.zeros([64]))
L4 = (tf.matmul(L3, W4) + B4)

W5 = tf.Variable(tf.random_normal([64,32]))
B5 = tf.Variable(tf.zeros([32]))
L5 = (tf.matmul(L4, W5) + B5)



W = tf.Variable(tf.random_normal([32,N_CLASSES]))#ÈšÖØ
b = tf.Variable(tf.zeros([N_CLASSES]))#Æ«ÖÃ
y = tf.matmul(L5, W) + b#Ô€²âÖµ

input_images = g1.get_tensor_by_name("input:0")
y_hat = tf.placeholder(tf.int64, [None])

loss = tf.losses.softmax_cross_entropy(
        tf.one_hot(y_hat, N_CLASSES), y,weights=1.0)

train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

print(y)

#%%
#正确率
pre = tf.argmax(y, 1)
correct_prediction = tf.equal(pre, y_hat)
evalution_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%

sess.run(tf.initializers.global_variables())
#%%

batch_s = 0
batch_t = BATCH



with g1.as_default():
    saver = tf.train.Saver()

    print('------------------start training-------------------')

    for i in range(STEPS):

        sess.run(train_step, feed_dict={
            input_images: train_images[batch_s:batch_t],
            y_hat: train_labels[batch_s:batch_t]
        })

        if i + 1 == STEPS or i % 50 == 0:
            sess.run(train_step, feed_dict={
                input_images: train_images[batch_s:batch_t],
                y_hat: train_labels[batch_s:batch_t]
            })


            saver.save(sess, TRAIN_FILE, global_step=i)

            upper = int(n_validation_set / 3.0)
            downer = 0
            whole_accuracy = 0.0

            for j in range(3):
                accuracy = sess.run(evalution_step, feed_dict={
                    input_images: validation_images[downer:upper],
                    y_hat: validation_labels[downer:upper]
                })
                whole_accuracy += accuracy

                downer = upper
                upper += int(n_validation_set / 3.0)
            print('Step %d: Validation accuracy = %.lf%%' % (i, (whole_accuracy / 3.0) * 100))

            upper = int(n_others / 3.0)
            downer = 0
            whole_accuracy = 0.0

            for j in range(3):
                accuracy = sess.run(evalution_step, feed_dict={
                    input_images: others_images[downer:upper],
                    y_hat: others_labels[downer:upper]
                })
                whole_accuracy += accuracy

                downer = upper
                upper += int(n_others / 3.0)
            print('Step %d: Other accuracy = %.lf%%' % (i, (whole_accuracy / 3.0) * 100))

        batch_s = batch_t
        if batch_s == n_train_set:
            batch_s = 0
        batch_t = batch_s + BATCH
        if batch_t > n_train_set:
            batch_t = n_train_set

    upper = int(n_test_set/3.0)
    downer = 0
    whole_accuracy = 0.0
    for j in range(3):
        accuracy = sess.run(evalution_step, feed_dict={
            input_images: test_images[downer:upper],
            y_hat: test_labels[downer:upper]
        })
        whole_accuracy += accuracy

        downer = upper
        upper += int(n_test_set / 3.0)
    print('----------------------------------------')
    print('After Step %d,the test accuracy = %.lf%%' % (STEPS, (whole_accuracy / 3.0) * 100))
    print('----------------------------------------')

    upper = int(n_others / 3.0)
    downer = 0
    whole_accuracy = 0.0

    for j in range(3):
        accuracy = sess.run(evalution_step, feed_dict={
            input_images: others_images[downer:upper],
            y_hat: others_labels[downer:upper]
        })
        whole_accuracy += accuracy

        downer = upper
        upper += int(n_others / 3.0)
    print('After Step %d,the other accuracy = %.lf%%' % (i, (whole_accuracy / 3.0) * 100))

    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[
        'add_5'])
    with tf.gfile.GFile(PB_FILE, mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
        f.write(output_graph_def.SerializeToString())



#%%
writer = tf.summary.FileWriter(TENSORBOARD_LOG, g1)
writer.close()
#%%
sess.close()
