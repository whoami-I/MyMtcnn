import os
import sys

import tensorflow as tf
import cv2 as cv
import numpy as np
from src.data_process.constants import *
from src.utils.look_img import *


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


def get_example(filename, label, info):
    label = int(label)
    im = cv.imread(filename)
    height, width, channel = im.shape
    im = im.tostring()
    example = dict()

    example['label'] = label
    example['img'] = im
    example['shape'] = [height, width, channel]

    offset = dict()
    offset['leftx'] = float(info[0])
    offset['lefty'] = float(info[1])
    offset['rightx'] = float(info[2])
    offset['righty'] = float(info[3])
    example['offset'] = offset

    landmark = dict()
    landmark['xlefteye'] = float(info[4])  # 初始化人脸坐标的值
    landmark['ylefteye'] = float(info[5])
    landmark['xrighteye'] = float(info[6])
    landmark['yrighteye'] = float(info[7])
    landmark['xnose'] = float(info[8])
    landmark['ynose'] = float(info[9])
    landmark['xleftmouth'] = float(info[10])
    landmark['yleftmouth'] = float(info[11])
    landmark['xrightmouth'] = float(info[12])
    landmark['yrightmouth'] = float(info[13])
    example['landmark'] = landmark
    return example


def generate_tfrecord(source_info, output_dir, tfrecord_file_name):
    """
    :param source_info: [[a,b],[a,b],...] a是图片文件夹，b是文件索引及点信息
    :param output_dir:  tfrecord文件 输出目录
    :param tfrecord_file_name: tfrecord文件名
    :return:
    """
    tfrecord_file_full_path = os.path.join(output_dir, tfrecord_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data_dir, index_file in source_info:

        with tf.io.TFRecordWriter(tfrecord_file_full_path) as tf_record_writer:

            with open(index_file) as index_fp:
                for line in index_fp:
                    if line.isspace(): continue
                    filename, label, *info = line.split(' ')
                    label = int(label)
                    if label == Const.LABEL_N:
                        # 只写入图像信息和label信息
                        info = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        instance = get_example(os.path.join(data_dir, filename), label, info)
                    elif label == Const.LABEL_POSI or label == Const.LABEL_PART:
                        # 写入图像，偏移量和label信息
                        info = list(map(float, info))
                        if len(info) == 4:
                            info.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            instance = get_example(os.path.join(data_dir, filename), label, info)
                        elif len(info) == 14:
                            instance = get_example(os.path.join(data_dir, filename), label, info)
                    bbox = instance['offset']
                    lm = instance['landmark']
                    roi = [bbox['leftx'], bbox['lefty'], bbox['rightx'], bbox['righty']]
                    landmark = [lm['xlefteye'], lm['ylefteye'], lm['xrighteye'], lm['yrighteye'], lm['xnose'],
                                lm['ynose'],
                                lm['xleftmouth'], lm['yleftmouth'], lm['xrightmouth'], lm['yrightmouth']]
                    # 此处的feature要注意，有些有s，有些没有s，不能写错，否则编译出错
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image/img': _bytes_feature(instance['img']),
                        'image/shape': _int64_feature(instance['shape']),
                        'image/label': _int64_feature(instance['label']),
                        'image/offset': _float_feature(roi),
                        'image/landmark': _float_feature(landmark)
                    }))
                    tf_record_writer.write(example.SerializeToString())

    pass


def decode_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    image_feature_description = {
        'image/img': tf.io.FixedLenFeature([], tf.string),
        'image/shape': tf.io.FixedLenFeature([3], tf.int64),
        'image/label': tf.io.FixedLenFeature([], tf.int64),
        'image/offset': tf.io.FixedLenFeature([4], tf.float32),
        'image/landmark': tf.io.FixedLenFeature([10], tf.float32),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    img_batch = []
    offset_batch = []
    label_batch = []
    landmark_batch = []
    parsed_dataset = dataset.map(_parse_image_function)
    for img_feature in parsed_dataset:
        img_raw = tf.io.decode_raw(img_feature['image/img'], tf.uint8)
        shape = tf.cast(img_feature['image/shape'], tf.int32)
        im = tf.reshape(img_raw, shape)
        label = tf.cast(img_feature['image/label'], tf.int32)
        img_batch.append(im)
        label_batch.append(label)
        if label == Const.LABEL_POSI or label == Const.LABEL_PART:
            offset = tf.cast(img_feature['image/offset'], tf.float32)
            landmark = tf.cast(img_feature['image/landmark'], tf.float32)
            offset_batch.append(offset)
            landmark_batch.append(landmark)

    return img_batch, label_batch, offset_batch, landmark_batch


if __name__ == '__main__':
    NET = Const.PNET
    if NET == Const.PNET:
        tfrecord = 'pnet_tfrecord'
    elif NET == Const.RNET:
        tfrecord = 'rnet_tfrecord'
    elif NET == Const.ONET:
        tfrecord = 'onet_tfrecord'
    else:
        raise Exception('net must be ONET,PNET,RNET')

    # img_dir = os.path.join(Const.root_path, Const.onet_positive_data_dir, Const.img_file_dir)
    # index_file = os.path.join(Const.root_path, Const.onet_positive_data_dir, Const.index_file_name)
    # generate_tfrecord([[img_dir, index_file]], Const.root_path, tfrecord)

    img_batch, label_batch, offset_batch, landmark_batch = decode_tfrecord(os.path
                                                                           .join(Const.root_path, tfrecord))
    for img in img_batch:
        im = np.array(img)
        print(im.shape)
        look_img(im)
