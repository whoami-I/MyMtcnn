import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from data_process.generate_p_net_tfrecord import decode_tfrecord
from src.data_process.constants import *
from src.net.Net import *


def train(model, tfrecord_file, epoch, save_model_file):
    dataset = decode_tfrecord(os.path.join(Const.root_path, tfrecord_file))
    for x in range(epoch + 1):
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        for i, (img, label, offset, landmark) in enumerate(dataset):
            # for index, imm in enumerate(img):
            # 归一化
            img = np.array(img, dtype=float)
            img = (img - 127.5) / 128.0
            with tf.GradientTape() as tape:
                cls_prob, bbox_pred = model(img)
                if len(cls_prob.shape)!=2:
                    cls_prob = tf.squeeze(cls_prob, [1, 2])
                    bbox_pred = tf.squeeze(bbox_pred, [1, 2])
                total_loss = classfier_loss(cls_prob, label) + 0.5 * offset_loss(bbox_pred, offset, label)

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if i % 300 == 0: print(total_loss)
        if x % 20 == 0: model.save(save_model_file, include_optimizer=True)
    model.save(save_model_file, include_optimizer=True)
    pass


if __name__ == '__main__':
    train(PNET(), Const.pnet_tfrecord, 120, 'p_model')
    train(RNET(), Const.rnet_tfrecord, 120, 'r_model')
    train(ONET(), Const.onet_tfrecord, 120, 'o_model')
