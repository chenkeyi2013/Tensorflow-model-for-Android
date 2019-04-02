import tensorflow as tf

#解析器格式
#features按照生成tfrecord文件时一样的定义
def parser(record):

    features = tf.parse_single_example(
    record,
    features = {
       'label':tf.FixedLenFeature([], tf.int64),
        'image_raw':tf.FixedLenFeature([], tf.string)
    })
    return features['label'], features['image_raw']


# filename:tfrecord文件路径
# buffer_size:打乱顺序，越大随机效果越好
def get_set(filename, buffer_size):
    input_files = [filename]
    DATASET = tf.data.TFRecordDataset(input_files)
    DATASET = DATASET.shuffle(buffer_size)
    DATASET = DATASET.map(parser)

    return DATASET


# 从数据集中加载数据
# size:抓取的Batch大小

def get_data(DATASET, size):

    iterator = DATASET.make_one_shot_iterator()
    feat1, feat2 = iterator.get_next()


    # 返回特征张量
    images = []
    labels = []

    with tf.Session() as sess:
        for i in range(size):
            #获取features
            f1, f2 = sess.run([feat1, feat2])
            # 解码byte
            image = tf.decode_raw(f2, tf.float32)

            image = tf.reshape(image, [224, 224, 3])

            images.append(image.eval())
            labels.append(f1)

    return images, labels
