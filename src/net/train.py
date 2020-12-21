import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from data_process.generate_p_net_tfrecord import decode_tfrecord
from src.data_process.constants import *
from src.net.Net import *
import time
import threading


def train(model, tfrecord_file, epoch, save_model_file):
    dataset = decode_tfrecord(os.path.join(Const.root_path, tfrecord_file))
    for x in range(epoch + 1):
        print('begin eporch:' + str(x))
        localtime = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime)

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        for i, (img, label, offset, landmark) in enumerate(dataset):
            # for index, imm in enumerate(img):
            # 归一化

            img = np.array(img, dtype=float)
            img = (img - 127.5) / 128.0
            with tf.GradientTape() as tape:
                cls_prob, bbox_pred = model(img)
                if len(cls_prob.shape) != 2:
                    cls_prob = tf.squeeze(cls_prob, [1, 2])
                    bbox_pred = tf.squeeze(bbox_pred, [1, 2])
                total_loss = classfier_loss(cls_prob, label) + 0.5 * offset_loss(bbox_pred, offset, label)

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if i % 300 == 0:
                    print(total_loss)
                    localtime = time.asctime(time.localtime(time.time()))
                    print("本地时间为 :", localtime)

        if x % 5 == 0: model.save(save_model_file, include_optimizer=True)
    model.save(save_model_file, include_optimizer=True)
    pass


def run_pnet():
    time.sleep(60 * 6)
    train(PNET(), Const.pnet_tfrecord, 25, 'p_model')


def run_rnet():
    time.sleep(60*3)
    train(RNET(), Const.rnet_tfrecord, 25, 'r_model')


def run_onet():
    train(ONET(), Const.onet_tfrecord, 25, 'o_model')


if __name__ == '__main__':
    # run_onet()
    # run_pnet()
    # run_rnet()
    p = threading.Thread(target=run_pnet)
    r = threading.Thread(target=run_rnet)
    o = threading.Thread(target=run_onet)
    p.start()
    r.start()
    o.start()

    p.join()
    r.join()
    o.join()
