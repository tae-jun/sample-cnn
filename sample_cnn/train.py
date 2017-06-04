import math
import os
import re

import tensorflow as tf

from glob import glob

from keras.callbacks import (TensorBoard, ModelCheckpoint,
                             EarlyStopping, CSVLogger)
from keras.optimizers import SGD

from sample_cnn.model import SampleCNN
from sample_cnn.ops import batch_inputs, evaluate

tf.logging.set_verbosity(tf.logging.INFO)

# Paths.
tf.flags.DEFINE_string('train_input_file_pattern', '',
                       'File pattern of TFRecord input files.')
tf.flags.DEFINE_string('val_input_file_pattern', '',
                       'File pattern of validation TFRecord input files.')
tf.flags.DEFINE_string('test_input_file_pattern', '',
                       'File pattern of test TFRecord input files.')
tf.flags.DEFINE_string('train_dir', '',
                       'Directory where to write event logs and checkpoints.')
tf.flags.DEFINE_string('checkpoint_prefix', 'best_weights',
                       'Prefix of the checkpoint filename.')

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
tf.flags.DEFINE_integer('n_test_examples', 4332,
                        'Number of examples in test dataset.')

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


def find_best_checkpoint(*dirs):
  best_checkpoint_path = None
  best_epoch = -1
  best_val_loss = 1e+10
  for dir in dirs:
    checkpoint_paths = glob('{}/{}*'.format(dir, FLAGS.checkpoint_prefix))
    for checkpoint_path in checkpoint_paths:
      epoch = int(re.findall('e\d+', checkpoint_path)[0][1:])
      val_loss = float(re.findall('l\d\.\d+', checkpoint_path)[0][1:])

      if val_loss < best_val_loss:
        best_checkpoint_path = checkpoint_path
        best_epoch = epoch
        best_val_loss = val_loss

  return best_checkpoint_path, best_epoch, best_val_loss


def train(initial_lr,
          stage_train_dir,
          checkpoint_path_to_load=None,
          initial_epoch=0):
  if not tf.gfile.Exists(stage_train_dir):
    tf.logging.info('Creating training directory: %s', stage_train_dir)
    tf.gfile.MakeDirs(stage_train_dir)

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

  # Create a model.
  model = SampleCNN(segments=x_train_batch,
                    val_segments=x_val_batch,
                    dropout_rate=FLAGS.dropout_rate)

  # Load weights from a checkpoint if exists.
  if checkpoint_path_to_load:
    print('Load weights from "{}".'.format(checkpoint_path_to_load))
    model.load_weights(checkpoint_path_to_load)

  # Setup an optimizer.
  optimizer = SGD(lr=initial_lr,
                  momentum=FLAGS.momentum,
                  decay=FLAGS.local_lr_decay,
                  nesterov=True)

  # Compile the model.
  model.compile_tfrecord(y=y_train_batch,
                         y_val=y_val_batch,
                         loss='binary_crossentropy',
                         optimizer=optimizer)

  # Setup a TensorBoard callback.
  tensor_board = TensorBoard(log_dir=stage_train_dir)

  # Use early stopping mechanism.
  early_stopping = EarlyStopping(monitor='val_loss', patience=FLAGS.patience)

  # Setup a checkpointer.
  checkpoint_path = make_path(
    stage_train_dir,
    FLAGS.checkpoint_prefix + '-e{epoch:03d}-l{val_loss:.4f}.hdf5')
  checkpointer = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True)

  # Setup a CSV logger.
  csv_logger = CSVLogger(filename=make_path(stage_train_dir, 'training.csv'),
                         append=True)

  # Kick-off the training!
  train_steps = calculate_steps(n_examples=FLAGS.n_train_examples,
                                n_segments=FLAGS.n_segments_per_audio,
                                batch_size=FLAGS.batch_size)
  val_steps = calculate_steps(n_examples=FLAGS.n_val_examples,
                              n_segments=FLAGS.n_segments_per_audio,
                              batch_size=FLAGS.batch_size)

  model.fit_tfrecord(epochs=100,
                     initial_epoch=initial_epoch,
                     steps_per_epoch=train_steps,
                     validation_steps=val_steps,
                     callbacks=[tensor_board, early_stopping,
                                checkpointer, csv_logger])

  # The end of the stage. Evaluate on test set.
  best_ckpt_path, *_ = find_best_checkpoint(stage_train_dir)
  print('The end of the stage. '
        'Start evaluation on test set using checkpoint "{}"'
        .format(best_ckpt_path))

  evaluate(input_file_pattern=FLAGS.test_input_file_pattern,
           weights_path=best_ckpt_path,
           n_examples=FLAGS.n_test_examples,
           n_audios_per_shard=FLAGS.n_audios_per_shard,
           print_progress=False)


def main(unused_argv):
  assert FLAGS.train_dir, '--train_dir is required'
  assert FLAGS.train_input_file_pattern, '--train_input_file_pattern is required'
  assert FLAGS.val_input_file_pattern, '--val_input_file_pattern is required'
  assert FLAGS.test_input_file_pattern, '--test_input_file_pattern is required'

  # Print all flags.
  print('@@ Flags')
  for key, value in FLAGS.__flags.items():
    print('{}={}'.format(key, value))

  for i in range(FLAGS.initial_stage, FLAGS.max_trains):
    stage_train_dir = make_path(FLAGS.train_dir, i)
    previous_stage_train_dir = make_path(FLAGS.train_dir, i - 1)
    next_stage_train_dir = make_path(FLAGS.train_dir, i + 1)

    # Pass if there is a training directory of the next stage.
    if os.path.isdir(next_stage_train_dir):
      continue

    # Setup the initial learning rate for the stage.
    decay = FLAGS.global_lr_decay ** i
    learning_rate = FLAGS.initial_learning_rate * decay

    # Create a directory for the stage.
    os.makedirs(stage_train_dir, exist_ok=True)

    # Find the best checkpoint to load weights.
    (ckpt_path, ckpt_epoch, ckpt_val_loss) = find_best_checkpoint(
      stage_train_dir, previous_stage_train_dir)

    print('\n@@ Start training stage {:02d}: lr={}, train_dir={}'
          .format(i, learning_rate, stage_train_dir))
    if ckpt_path:
      print('Found a trained model: epoch={}, val_loss={}, path={}'
            .format(ckpt_epoch, ckpt_val_loss, ckpt_path))
    else:
      print('No trained model found.')

    train(initial_lr=learning_rate,
          stage_train_dir=stage_train_dir,
          checkpoint_path_to_load=ckpt_path,
          initial_epoch=ckpt_epoch + 1)

  print('\nDone.')


if __name__ == '__main__':
  tf.app.run()
