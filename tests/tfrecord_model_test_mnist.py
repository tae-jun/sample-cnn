import os

import tensorflow as tf
import numpy as np

from keras.layers import Input, Conv2D, Dense, Flatten
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint

from sample_cnn.keras_utils.tfrecord_model import TFRecordModel


def data_to_tfrecord(images, labels, filename):
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  """ Save data into TFRecord """
  if not os.path.isfile(filename):
    num_examples = images.shape[0]

    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
      writer.write(example.SerializeToString())
    writer.close()
  else:
    print('tfrecord already exist')


def read_and_decode(filename, one_hot=True, n_classes=None):
  """ Return tensor to read from TFRecord """
  filename_queue = tf.train.string_input_producer([filename])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example,
                                     features={
                                       'label': tf.FixedLenFeature([],
                                                                   tf.int64),
                                       'image_raw': tf.FixedLenFeature([],
                                                                       tf.string),
                                     })
  # You can do more image distortion here for training data
  img = tf.decode_raw(features['image_raw'], tf.uint8)
  img.set_shape([28 * 28])
  img = tf.reshape(img, [28, 28, 1])
  img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

  label = tf.cast(features['label'], tf.int32)
  if one_hot and n_classes:
    label = tf.one_hot(label, n_classes)

  return img, label


def build_net(inputs):
  net = Conv2D(32, 3, strides=2, activation='relu')(inputs)
  net = Conv2D(32, 3, strides=2, activation='relu')(net)
  net = Flatten()(net)
  net = Dense(128, activation='relu')(net)
  net = Dense(10, activation='softmax')(net)
  return net


if __name__ == '__main__':
  TMP_DIR = '_test_tmp_dir_'
  TRAIN_TFRECORD_PATH = TMP_DIR + '/mnist_train.tfrecords'
  VAL_TFRECORD_PATH = TMP_DIR + '/mnist_val.tfrecords'
  TEST_TFRECORD_PATH = TMP_DIR + '/mnist_test.tfrecords'
  WEIGHTS_PATH = TMP_DIR + '/weights.hdf5'

  batch_size = 64
  n_train = 50000

  # Create temp dir
  os.makedirs(TMP_DIR, exist_ok=True)

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)

  x_val = x_train[n_train:]
  y_val = y_train[n_train:]

  x_train = x_train[:n_train]
  y_train = y_train[:n_train]

  data_to_tfrecord(images=x_train,
                   labels=y_train,
                   filename=TRAIN_TFRECORD_PATH)
  data_to_tfrecord(images=x_val,
                   labels=y_val,
                   filename=VAL_TFRECORD_PATH)
  data_to_tfrecord(images=x_test,
                   labels=y_test,
                   filename=TEST_TFRECORD_PATH)

  x_train, y_train = read_and_decode(filename=TRAIN_TFRECORD_PATH,
                                     one_hot=True,
                                     n_classes=10)
  x_val, y_val = read_and_decode(filename=VAL_TFRECORD_PATH,
                                 one_hot=True,
                                 n_classes=10)
  x_test, y_test = read_and_decode(filename=TEST_TFRECORD_PATH,
                                   one_hot=True,
                                   n_classes=10)

  x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train, y_train],
                                                        batch_size=batch_size,
                                                        capacity=2000,
                                                        min_after_dequeue=1000,
                                                        name='train_batch')
  x_val_batch, y_val_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=batch_size,
                                                    capacity=2000,
                                                    min_after_dequeue=1000,
                                                    name='val_batch')
  x_test_batch, y_test_batch = tf.train.shuffle_batch([x_test, y_test],
                                                      batch_size=batch_size,
                                                      capacity=2000,
                                                      min_after_dequeue=1000,
                                                      name='test_batch')

  x_train_input = Input(tensor=x_train_batch)
  x_val_input = Input(tensor=x_val_batch)
  x_test_input = Input(tensor=x_test_batch)
  logits = build_net(x_train_input)

  model = TFRecordModel(inputs=x_train_input,
                        outputs=logits,
                        val_inputs=x_val_input)

  model.compile_tfrecord(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'],
                         y=y_train_batch,
                         y_val=y_val_batch)

  model.summary()

  model_checkpoint = ModelCheckpoint(
    filepath=TMP_DIR + '/best_weights.{epoch:02d}-{val_loss:.4f}.hdf5',
    save_best_only=True)

  model.fit_tfrecord(epochs=10,
                     verbose=2,
                     steps_per_epoch=n_train // batch_size + 1,
                     validation_steps=10000 // batch_size + 1,
                     callbacks=[model_checkpoint])

  model.save_weights(WEIGHTS_PATH)
  model.load_weights(WEIGHTS_PATH)

  test_outputs = model(x_test_input)
  test_model = TFRecordModel(inputs=x_test_input,
                             outputs=test_outputs)
  test_model.compile_tfrecord(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'],
                              y=y_test_batch)

  loss, accuracy = test_model.evaluate_tfrecord(steps=10000 // batch_size + 1)

  print('accuracy={}, loss={}'.format(accuracy, loss))
  print('Done.')
