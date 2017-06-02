import os
import math

import tensorflow as tf

from keras.optimizers import SGD
from keras.callbacks import (TensorBoard, ModelCheckpoint, EarlyStopping,
                             CSVLogger)

from sample_cnn.model import SampleCNN
from sample_cnn.ops.inputs import batch_inputs

tf.logging.set_verbosity(tf.logging.INFO)

# Paths.
tf.flags.DEFINE_string('train_input_file_pattern', '',
                       'File pattern of TFRecord input files.')
tf.flags.DEFINE_string('val_input_file_pattern', '',
                       'File pattern of validation TFRecord input files.')
tf.flags.DEFINE_string('train_dir', '',
                       'Directory where to write event logs and checkpoints.')
tf.flags.DEFINE_string('best_weights_filename', 'best_weights.hdf5',
                       'Filename of weights of the best model.')

# Batch options.
tf.flags.DEFINE_integer('batch_size', 23, 'Batch size.')
tf.flags.DEFINE_integer('n_audios_per_shard', 100,
                        'Number of audios per shard.')
tf.flags.DEFINE_integer('n_segments_per_audio', 10,
                        'Number of segments per audio.')
tf.flags.DEFINE_integer('n_train_examples', 15250,
                        'Number of examples in training dataset.')
tf.flags.DEFINE_integer('n_val_examples', 1529,
                        'Number of examples in validation dataset.')

tf.flags.DEFINE_integer('n_read_threads', 4, 'Number of example reader.')

# Learning options.
tf.flags.DEFINE_float('initial_learning_rate', 0.01, 'Initial learning rate.')
tf.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.flags.DEFINE_float('dropout_rate', 0.5, 'Dropout keep probability.')
tf.flags.DEFINE_float('global_lr_decay', 0.2, 'Global learning rate decay.')
tf.flags.DEFINE_float('local_lr_decay', 1e-6, 'Local learning rate decay.')

# Training options.
tf.flags.DEFINE_integer('patience', 3, 'A patience for the early stopping.')
tf.flags.DEFINE_integer('max_trains', 5, 'Number of re-training.')
tf.flags.DEFINE_integer('initial_stage', 0,
                        'The stage where to start training.')

FLAGS = tf.app.flags.FLAGS


def make_path(*paths):
  path = os.path.join(*[str(path) for path in paths])
  path = os.path.realpath(path)
  return path


def calculate_steps(n_examples, n_segments, batch_size):
  steps = 1. * n_examples * n_segments
  steps = math.ceil(steps / batch_size)
  return steps


def train(learning_rate, train_dir, past_best_weight_path):
  if not tf.gfile.Exists(train_dir):
    tf.logging.info('Creating training directory: %s', train_dir)
    tf.gfile.MakeDirs(train_dir)

  x_train_batch, y_train_batch = batch_inputs(
    file_pattern=FLAGS.train_input_file_pattern,
    batch_size=FLAGS.batch_size,
    is_training=True,
    n_read_threads=FLAGS.n_read_threads,
    examples_per_shard=FLAGS.n_audios_per_shard * FLAGS.n_segments_per_audio,
    shard_queue_name='train_filename_queue',
    example_queue_name='train_input_queue')

  x_val_batch, y_val_batch = batch_inputs(
    file_pattern=FLAGS.val_input_file_pattern,
    batch_size=FLAGS.batch_size,
    is_training=False,
    n_read_threads=1,
    examples_per_shard=FLAGS.n_audios_per_shard * FLAGS.n_segments_per_audio,
    shard_queue_name='val_filename_queue',
    example_queue_name='val_input_queue')

  model = SampleCNN(segments=x_train_batch,
                    dropout_rate=FLAGS.dropout_rate)

  if past_best_weight_path:
    print('Load weights from "{}".'.format(past_best_weight_path))
    model.load_weights(past_best_weight_path)

  optimizer = SGD(lr=learning_rate,
                  momentum=FLAGS.momentum,
                  decay=FLAGS.local_lr_decay,
                  nesterov=True)
  model.compile_tfrecord(y_batch=y_train_batch,
                         loss='binary_crossentropy',
                         optimizer=optimizer)

  best_weights_path = make_path(train_dir, FLAGS.best_weights_filename)

  # Setup callbacks.
  tensor_board = TensorBoard(log_dir=train_dir)
  early_stopping = EarlyStopping(monitor='val_loss', patience=FLAGS.patience)
  model_checkpoint = ModelCheckpoint(
    filepath=best_weights_path,
    monitor='val_loss',
    save_best_only=True)
  csv_logger = CSVLogger(filename=make_path(train_dir, 'training.csv'),
                         append=True)

  # Kick-off the training!
  train_steps = calculate_steps(n_examples=FLAGS.n_train_examples,
                                n_segments=FLAGS.n_segments_per_audio,
                                batch_size=FLAGS.batch_size)
  val_steps = calculate_steps(n_examples=FLAGS.n_val_examples,
                              n_segments=FLAGS.n_segments_per_audio,
                              batch_size=FLAGS.batch_size)

  model.fit_tfrecord(steps_per_epoch=train_steps,
                     epochs=1000,
                     callbacks=[tensor_board, early_stopping,
                                model_checkpoint, csv_logger],
                     validation_batch=(x_val_batch, y_val_batch),
                     validation_steps=val_steps)

  # TODO: Evaluate on test set.


def main(unused_argv):
  assert FLAGS.train_input_file_pattern, '--train_input_file_pattern is required'
  assert FLAGS.train_dir, '--train_dir is required'
  assert FLAGS.val_input_file_pattern, '--val_input_file_pattern is required'

  # Print all flags.
  print('### Flags')
  for key, value in FLAGS.__flags.items():
    print('{}={}'.format(key, value))

  best_weights_path = None
  for i in range(FLAGS.initial_stage, FLAGS.max_trains):
    if os.path.isdir(make_path(FLAGS.train_dir, i + 1)):
      continue

    decay = FLAGS.global_lr_decay ** i
    learning_rate = FLAGS.initial_learning_rate * decay

    train_dir = make_path(FLAGS.train_dir, i)
    os.makedirs(train_dir, exist_ok=True)

    current_weights_path = make_path(train_dir, FLAGS.best_weights_filename)
    past_weights_path = make_path(FLAGS.train_dir, i - 1,
                                  FLAGS.best_weights_filename)
    if os.path.isfile(current_weights_path):
      best_weights_path = current_weights_path
    elif os.path.isfile(past_weights_path):
      best_weights_path = past_weights_path

    print('\n### Start training stage {}'.format(i))
    print('learning_rate={}'.format(learning_rate))
    print('train_dir={}\n'.format(train_dir))

    train(learning_rate, train_dir, best_weights_path)

    best_weights_path = make_path(train_dir, FLAGS.best_weights_filename)


if __name__ == '__main__':
  tf.app.run()
