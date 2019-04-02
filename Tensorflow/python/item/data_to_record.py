#%%
import tensorflow as tf
import glob
import os.path
import os
import numpy as np
#%%
#忽略CPU警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#原始输入数据的目录
INPUT_DATA = "/home/chenkeyi/Workplace/google/train_data/asd"
filename_test = "/home/chenkeyi/Workplace/google/tfrecords/test.tfrecords"
filename_train = "/home/chenkeyi/Workplace/google/tfrecords/train.tfrecords"
filename_validation = "/home/chenkeyi/Workplace/google/tfrecords/validation.tfrecords"
#%%
#各个数据集
training_images = []
training_labels = []
testing_images = []
testing_labels = []
validation_images = []
validation_labels = []

#划分比例
TEST_PERCENTAGE = 5
VALIDATION_PERCENTAGE = 5

files = [filename_train, filename_test, filename_validation]
data_list = [training_images, testing_images, validation_images]
label_list = [training_labels, testing_labels, validation_labels]
#%%
sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
is_root_dir = True

current_label = 0
# 读取所有子目录
for sub_dir in sub_dirs:
    if is_root_dir:
        is_root_dir = False

        continue
    # 获取子目录下的图片文件
    extensions = 'jpg'

    # file_list存储所有文件的绝对路径
    file_list = []

    file_glob = os.path.join(INPUT_DATA, sub_dir, '*.' + extensions)
    if file_glob:
        file_list.extend(glob.glob(file_glob))

    times = 1

    for file_name in file_list:
        # 读取图片
        image_raw_data = tf.gfile.GFile(file_name, 'rb').read()
        # 解码图片
        image = tf.image.decode_jpeg(image_raw_data, channels=3)
        # 统一图片格式---------最杀时间的一步,基本全部要转换

        if image.dtype != tf.float32:
            # 转成实数型
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # image类型为Tensor
        image = tf.image.resize_images(image, [224, 224])
        # 将image转为矩阵
        sess = tf.Session()
        image_value = sess.run(image)

        # 划分数据集
        chance = np.random.randint(100)
        if chance < VALIDATION_PERCENTAGE:
            validation_images.append(image_value)
            validation_labels.append(current_label)
        elif chance < TEST_PERCENTAGE + VALIDATION_PERCENTAGE:
            testing_images.append(image_value)
            testing_labels.append(current_label)
        else:
            training_images.append(image_value)
            training_labels.append(current_label)
        sess.close()

        # 等的太久了 看一下进度:
        print("DATA %d : %d images is done" % (current_label, times))
        times += 1

    current_label += 1
#%%
print(len(training_images))
print(len(validation_images))
print(len(testing_images))
#%%
with tf.Session() as sess:
    for i in range(3):
        writer = tf.python_io.TFRecordWriter(files[i])
        for index in range(len(data_list[i])):
            images_raw = data_list[i][index].tostring()
            example = tf.train.Example(features = tf.train.Features(feature = {
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[label_list[i][index]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images_raw]))
            }))
            writer.write(example.SerializeToString())
            print("%d : %d images is done" % (i, index))
        writer.close()