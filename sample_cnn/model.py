from keras.layers import (Conv1D, MaxPool1D, BatchNormalization,
                          Dense, Dropout, Activation, Flatten, Reshape)

from sample_cnn.keras_utils.tfrecord_model import TFRecordModel


class SampleCNN(TFRecordModel):
  def __init__(self, inputs,
               n_outputs=50,
               activation='relu',
               kernel_initializer='he_uniform',
               dropout_rate=0.5):
    # 59049
    net = Reshape([-1, 1])(inputs)
    # 59049 X 1
    net = Conv1D(128, 3, strides=3, padding='valid',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    # 19683 X 128
    net = Conv1D(128, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 6561 X 128
    net = Conv1D(128, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 2187 X 128
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 729 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 243 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 81 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 27 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 9 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 3 X 256
    net = Conv1D(512, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 1 X 512
    net = Conv1D(512, 1, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    # 1 X 512
    net = Dropout(dropout_rate)(net)
    net = Flatten()(net)

    outputs = Dense(units=n_outputs, activation='sigmoid')(net)

    super(SampleCNN, self).__init__(inputs=inputs, outputs=outputs)
