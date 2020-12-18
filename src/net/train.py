import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from data_process.generate_p_net_tfrecord import decode_tfrecord
from src.data_process.constants import *
from src.net.Net import *


def train(model, tfrecord_file):
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    dataset = decode_tfrecord(os.path.join(Const.root_path, tfrecord_file))
    for i, (img, label, offset, landmark) in enumerate(dataset):
        # for index, imm in enumerate(img):

        with tf.GradientTape() as tape:
            cls_prob, bbox_pred = model(img)
            print(cls_prob.shape, bbox_pred.shape)
            cls_prob = tf.squeeze(cls_prob, [1, 2])
            bbox_pred = tf.squeeze(bbox_pred, [1, 2])
            total_loss = classfier_loss(cls_prob, label) + 0.5 * offset_loss(bbox_pred, offset, label)
            print(total_loss)
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    pass


if __name__ == '__main__':
    train(PNET(), Const.pnet_tfrecord)
