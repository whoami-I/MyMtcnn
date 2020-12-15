import tensorflow as tf
import tensorflow.keras as keras
from src.data_process.constants import *


def PNET():
    input = keras.Input([None, None, 3])
    x = keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1),
                            kernel_regularizer=keras.regularizers.l2(0.0005))(input)
    x = keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
                            kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                            kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])(x)

    classfier = keras.layers.Conv2D(filters=2, kernel_size=(1, 1),
                                    activation=keras.activations.softmax)(x)
    offset = keras.layers.Conv2D(filters=4, kernel_size=(1, 1))(x)
    # print(x, classfier_prob, offset_prob)
    model = keras.models.Model([input], [classfier, offset])
    classfier_prob = tf.squeeze(classfier, axis=[1, 2])
    offset_prob = tf.squeeze(offset, axis=[1, 2])

    return model


def classfier_loss(classfier_prob, label):
    """
    :param classfier_prob: (n,2),类似于[[不是人的概率，是人的概率],
                                        不是人的概率，是人的概率],
                                        不是人的概率，是人的概率],
                                        .......
                                        .......]

    :param label: (n,1)
    :return:
    此损失函数计算part和positive和negative是否判断正确的交叉熵，假设网络判断
    图片不是人和是人的概率为[a,b],那么对于part和positive的交叉熵为：-[log(1-a)+log(b)],
    因为1-a=b,所以交叉熵为-2*log(b),同理对于negative的交叉熵为-2*log(a)
    """
    if classfier_prob.shape[0] != label.shape[0]:
        raise Exception('different row in classfier_loss')
    num_row = classfier_prob.shape[0]
    label = tf.reshape(label, [-1])
    total_loss = 0.0
    for val, index in enumerate(label):
        if val == Const.LABEL_POSI or val == Const.LABEL_PART:
            total_loss = total_loss - 2 * tf.math.log(classfier_prob[index][1] + 1e-10)
        elif (val == Const.LABEL_N):
            total_loss = total_loss - 2 * tf.math.log(classfier_prob[index][0] + 1e-10)
        else:
            raise Exception('label is none of p,n,part')

    return total_loss / num_row


def offset_loss(offset_probe, offset_target, label):
    """
    :param offset_probe: (n,4)
    :param offset_target:(n,4)
    :param label: (n,1)
    :return:
    """
    if offset_probe.shape[0] != label.shape[0] or offset_target.shape[0] != label.shape[0]:
        raise Exception('different row in classfier_loss')
    num_row = offset_probe.shape[0]
    label = tf.reshape(label, [-1])
    total_loss = 0.0
    for val, index in enumerate(label):
        if val == Const.LABEL_POSI or val == Const.LABEL_PART:
            loss = offset_probe[index] - offset_target[index]
            loss = tf.math.square(loss)
            total_loss = total_loss - tf.reduce_sum(loss)
        elif val == Const.LABEL_N:
            pass  # 当为负例时，没有边框损失
            # total_loss = total_loss - 2 * tf.math.log(classfier_prob[index][0] + 1e-10)
        else:
            raise Exception('label is none of p,n,part')

    return total_loss / num_row


if __name__ == '__main__':
    PNET()
