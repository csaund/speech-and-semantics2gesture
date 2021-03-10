import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from my_audio_to_pose_net.my_keras_model import MyKerasModel
import numpy as np


class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        # self.seq = get_simple_convlstm()
        self.convlstm = get_simple_convlstm()
        # self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False, **kwargs):
        return self.convlstm(inputs)


class SimpleConvLstm(MyKerasModel):
    def __init__(self, hparams, batch_size):
        super().__init__(hparams, batch_size)
        self.convlstm = get_simple_convlstm()

    def call(self, dataset_element, training=False, **kwargs):
        return self.convlstm(dataset_element['input'])

    def compute_loss(self, dataset_element, outputs):
        # return {'loss': tf.reduce_sum(tf.abs(dataset_element['output'] - outputs))}
        return {'loss': tf.reduce_sum(tf.losses.binary_crossentropy(dataset_element['output'], outputs))}


def get_simple_convlstm():
    seq = keras.Sequential(
        [
            keras.Input(
                shape=(None, 40, 40, 1)
            ),  # Variable-length sequence of 40x40x1 frames
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
            ),
        ]
    )

    # seq.compile(loss="binary_crossentropy", optimizer="adadelta")
    return seq



