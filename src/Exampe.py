import os
import random
import sys

import tensorflow as tf
import cv2
from PIL import Image

def _int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



def _process_image_withoutcoder(filename):
    """
    利用cv2将filename指向的图片tostring
    """

    image = cv2.imread(filename)

    # transform data into string format
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    # return string data and initial height and width of the image
    return image_data, height, width


def _convert_to_example_simple(image_example, image_buffer):
    """
        covert to tfrecord file
    Parameter
    ------------
        image_example: dict, an image example
        image_buffer: string, JPEG encoding of RGB image
    Return
    -----------
        Example proto
    """
    class_label = image_example['label']

    bbox = image_example['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    # landmark = [bbox['xlefteye'],bbox['ylefteye'],bbox['xrighteye'],bbox['yrighteye'],bbox['xnose'],bbox['ynose'],
    #             bbox['xleftmouth'],bbox['yleftmouth'],bbox['xrightmouth'],bbox['yrightmouth']]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        # 'image/landmark': _float_feature(landmark)
    }))
    return example


# 从图片和注释文件里加载数据并将其添加到TFRecord里
# 参数（变量）：filename:存有数据的字典；tfrecord_writer:用来写入TFRecord的writer

def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    # print('---', filename)

    # imaga_data:转化为字符串的图片
    # height:图片原始高度
    # width:图片原始宽度
    # image_example：包含图片信息的字典
    # print(filename)
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())  # 将imaga_data转化到image_example中并写入tfrecord


def _get_output_filename(output_dir,net):
    # 定义一下输出的文件名

    # return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    # st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # time.strftime() 函数接收以时间元组，并返回以可读字符串表示的当地时间，格式由参数format决定:time.strftime(format[, t]),用来输出当前时间
    # 返回的是'../../DATA/imglists/PNet/train_PNet_landmark.tfrecord'
    return '%s/train_%s_landmark.tfrecord' % (output_dir,net)


def run(dataset_dir,net,output_dir,shuffling=False):
    """
    运行转换操作
    Args:
      dataset_dir: 数据集所在的数据集目录
      output_dir: 输出目录
    """

    # tfrecord name
    tf_filename = _get_output_filename(output_dir,net)  # '../../DATA/imglists/PNet/train_PNet_landmark.tfrecord'

    if tf.io.gfile.exists(tf_filename):  # tf.io.gfile模块提供了文件操作的API,包括文件的读取、写入、删除、复制等等
        print('Dataset files already exist. Exiting without re-creating them.')  # 判断是否存在同名文件
        return

    # 获得数据集，并打乱顺序
    dataset = get_dataset(dataset_dir)
    print(dataset)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        # random.seed(12345454)
        random.shuffle(dataset)  # 打乱dataset数据集的顺序

    # Process dataset files.
    # write the data to tfrecord
    print('lala')
    with tf.io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):  # 读取dataset的索引和内容
            if (i + 1) % 1 == 0:
                sys.stdout.write('\r>> %d/%d images has been converted' % (
                i + 1, len(dataset)))  # 输出“x00/ len(dataset) images has been converted”
            sys.stdout.flush()  # 以一定间隔时间刷新输出
            filename = image_example['filename']  # 赋值
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # 最后，编写标签文件
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir):
    # 获取文件名字，标签和注释
    item = 'label-train%s.txt'%(dir)

    dataset_dir = os.path.join(dir, item)  # dataset_dir = '../../DATA/imglists/PNet/train_PNet_landmark.txt'
    # print(dataset_dir)
    imagelist = open(dataset_dir, 'r')  # 以只读的形式打开train_PNet_landmark.txt，并传入imagelist里面

    dataset = []  # 新建列表
    for line in imagelist.readlines():  # 按行读取imagelist里面的内容
        info = line.strip().split(' ')  # .strip().split()去除每一行首尾空格并且以空格为分隔符读取内容到info里面
        data_example = dict()  # 新建字典
        bbox = dict()
        data_example['filename'] = info[0]  # filename=info[0]
        # print(data_example['filename'])
        data_example['label'] = int(info[1])  # label=info[1]，info[1]的值有四种可能，1，0，-1，-2；分别对应着正、负、无关、关键点样本
        bbox['xmin'] = 0  # 初始化bounding box的值
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        # bbox['xlefteye'] = 0  # 初始化人脸坐标的值
        # bbox['ylefteye'] = 0
        # bbox['xrighteye'] = 0
        # bbox['yrighteye'] = 0
        # bbox['xnose'] = 0
        # bbox['ynose'] = 0
        # bbox['xleftmouth'] = 0
        # bbox['yleftmouth'] = 0
        # bbox['xrightmouth'] = 0
        # bbox['yrightmouth'] = 0
        if len(info) == 6:  # 当info的长度等于6时，表示此时的info是正样本或者无关样本
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        # if len(info) == 12:  # 当info的长度等于12时，表示此时的info是landmark样本
        #     bbox['xlefteye'] = float(info[2])
        #     bbox['ylefteye'] = float(info[3])
        #     bbox['xrighteye'] = float(info[4])
        #     bbox['yrighteye'] = float(info[5])
        #     bbox['xnose'] = float(info[6])
        #     bbox['ynose'] = float(info[7])
        #     bbox['xleftmouth'] = float(info[8])
        #     bbox['yleftmouth'] = float(info[9])
        #     bbox['xrightmouth'] = float(info[10])
        #     bbox['yrightmouth'] = float(info[11])

        data_example['bbox'] = bbox  # 将bounding box值传入字典
        dataset.append(data_example)  # 将data_example字典内容传入列表dataset

    return dataset  # 返回的是dataset，datase是个列表，但里面每个元素都是一个字典，每个字典都含有3个key，分别是filename、label和bounding box


if __name__ == '__main__':
    dir = '12'
    net = 'PNet'
    output_directory = '12'
    run(dir,net,output_directory,shuffling=True)
