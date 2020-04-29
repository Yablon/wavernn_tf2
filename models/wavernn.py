import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


class WaveRNN(tf.keras.layers.Layer):

  def __init__(self, config, name='WaveRNN'):
    super().__init__(name=name)
    self.hidden_size = config.hidden_size
    self.split_size = config.hidden_size // 2
    quantisation = config.quantisation

    self.R = Dense(3 * self.hidden_size, use_bias=False)
    self.O1 = Dense(self.split_size, activations='relu')
    self.O2 = Dense(quantisation)
    self.O3 = Dense(self.split_size, activations='relu')
    self.O4 = Dense(quantisation)

    self.I_coarse = Dense(2, 3 * self.split_size, use_bias=False)
    self.I_fine = Dense(3, 3 * self.split_size, use_bias=False)


  def build(self, input_shape):
    self.bias_u = self.add_weight(name='bias_u',
                                  shape=(self.hidden_size,),
                                  initializer=tf.zeros_initializer,
                                  trainable=True)
    self.bias_r = self.add_weight(name='bias_r',
                                  shape=(self.hidden_size,),
                                  initializer=tf.zeros_initializer,
                                  trainable=True)
    self.bias_e = self.add_weight(name='bias_e',
                                  shape=(self.hidden_size,),
                                  initializer=tf.zeros_initializer,
                                  trainable=True)
    self.builit = True


  def call(self, inputs):
    prev_y = inputs[0]
    prev_hidden = inputs[1]
    current_coarse = inputs[2]

    R_hidden = self.R(prev_hidden)
    R_u, R_r, R_e = tf.split(R_hidden, 3, axis=-1)

    coarse_input_proj = self.I_coarse(prev_y)
    I_coarse_u, I_coarse_r, I_coarse_e = tf.split(coarse_input_proj, 3, axis=-1)
    fine_input = tf.concat([prev_y, current_coarse], axis=-1)
    fine_input_proj = self.I_fine(fine_input)
    I_fine_u, I_fine_r, I_fine_e = tf.split(fine_input_proj, 3, axis=-1)

    I_u = tf.concat(I_coarse_u, I_fine_u)
    I_r = tf.concat(I_coarse_r, I_fine_r)
    I_e = tf.concat(I_coarse_e, I_fine_e)

    u = tf.keras.backend.sigmoid(R_u + I_u + self.bias_u)
    r = tf.keras.backend.sigmoid(R_r + I_r + self.bias_r)
    e = tf.keras.backend.tanh(r* R_e + I_e + self.bias_e)
    hidden = u * prev_hidden + (1. - u) * e

    hidden_coarse, hidden_fine = tf.split(hidden, 2, axis=-1)

    out_coarse = self.O2(self.O1(hidden_coarse))
    out_fine = self.O4(self.O3(hidden_fine))
    return out_coarse, out_fine, hidden
