# kfdummy.py
# KF 12/10/2018
import tensorflow as tf

class KFDummy:
    @staticmethod
    def build(input_width, input_height, input_channel, label_dim):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=((input_width, input_height, input_channel))),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(label_dim, activation=tf.nn.softmax)
            ])

        return model 
