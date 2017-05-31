import os
import json

import tensorflow as tf
import tensorflow.contrib.slim as slim

from sample_cnn.model.sample_cnn import SampleCNN
from sample_cnn.configuration import ModelConfig
from sample_cnn.ops.inputs import batch_inputs
from sample_cnn.ops.evaluation import evaluate, inference

tf.logging.set_verbosity(tf.logging.INFO)

# Paths.
tf.flags.DEFINE_string('train_input_file_pattern', '',
                       'File pattern of TFRecord input files.')
tf.flags.DEFINE_string('val_input_file_pattern', '',
                       'File pattern of validation TFRecord input files.')
tf.flags.DEFINE_string('train_dir', '',
                       'Directory where to write event logs and checkpoints.')
tf.flags.DEFINE_string('best_ckpt_name', 'best',
                       'Checkpoint name of the best model.')

# Batch options.
tf.flags.DEFINE_integer('batch_size', 23, 'Batch size.')
tf.flags.DEFINE_integer('n_audios_per_shard', 100,
                        'Number of audios per shard.')
tf.flags.DEFINE_integer('n_val_examples', 1529,
                        'Number of examples in validation dataset.')

tf.flags.DEFINE_integer('n_read_threads', 4, 'Number of example reader.')

# Learning options.
tf.flags.DEFINE_float('initial_learning_rate', 0.01, 'Initial learning rate.')
tf.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability.')
tf.flags.DEFINE_float('global_lr_decay', 0.2, 'Global learning rate decay.')
tf.flags.DEFINE_float('local_lr_decay', 1e-6, 'Local learning rate decay.')

# Training options.
tf.flags.DEFINE_integer('validation_every_n_steps', 6630,
                        'Number of steps to validate the model periodically.'
                        '(default 1 epoch)')
tf.flags.DEFINE_integer('patience', 3,
                        'If the validation loss does not decrease even after '
                        'learning `patience * validation_every_n_steps` more '
                        'steps, decay learning rate.')
tf.flags.DEFINE_integer('max_trains', 5, 'Number of re-training.')
tf.flags.DEFINE_integer('max_steps', 10000000, 'Number of batches to run.')
tf.flags.DEFINE_integer('log_every_n_steps', 10,
                        'Number of steps to log loss periodically.')
tf.flags.DEFINE_integer('max_checkpoints_to_keep', 1,
                        'Maximum number of checkpoints to keep.')

FLAGS = tf.app.flags.FLAGS


def _join_and_norm_path(*paths):
  path = os.path.join(*[str(path) for path in paths])
  path = os.path.realpath(path)
  return path


def _save_best_scores(train_dir, loss, roc_auc, filename='best_scores.json'):
  scores = {
    'loss': loss,
    'roc_auc': roc_auc
  }

  with open(_join_and_norm_path(train_dir, filename), 'w') as f:
    json.dump(scores, f)


def _load_best_scores(train_dir, filename='best_scores.json'):
  path = _join_and_norm_path(train_dir, filename)

  if not os.path.isfile(path):
    return None, None

  with open(path) as f:
    data = json.load(f)

  return data['loss'], data['roc_auc']


def _train(learning_rate, train_dir, previous_train_dir):
  # Print all flags.
  for key, value in FLAGS.__flags.items():
    print(key, '=', value)

  if not tf.gfile.Exists(train_dir):
    tf.logging.info('Creating training directory: %s', train_dir)
    tf.gfile.MakeDirs(train_dir)

  best_ckpt_path = _join_and_norm_path(train_dir, FLAGS.best_ckpt_name)
  best_ckpt_latest_filename = FLAGS.best_ckpt_name + '_checkpoint'

  # Set up the default model configurations.
  config = ModelConfig(mode='train')
  config.dropout_keep_prob = FLAGS.dropout_keep_prob

  g = tf.Graph()
  with g.as_default():
    # Batch.
    segment, labels = batch_inputs(
      file_pattern=FLAGS.train_input_file_pattern,
      batch_size=FLAGS.batch_size,
      is_training=True,
      is_sequence=False,
      n_read_threads=FLAGS.n_read_threads,
      examples_per_shard=FLAGS.n_audios_per_shard * 10,
      shard_queue_name='train_filename_queue',
      example_queue_name='train_input_queue')

    # sequence: [batch_size, 10, 58081]
    val_sequence, val_labels = batch_inputs(
      file_pattern=FLAGS.val_input_file_pattern,
      batch_size=FLAGS.batch_size,
      is_training=False,
      is_sequence=True,
      n_read_threads=1,
      examples_per_shard=FLAGS.n_audios_per_shard,
      shard_queue_name='val_filename_queue',
      example_queue_name='val_input_queue')

    # Build model.
    scnn = SampleCNN(config)
    logits = scnn(segment)
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    # Validation.
    val_pred, val_loss = inference(val_sequence, val_labels)

    # Global step.
    global_step = slim.create_global_step()

    # Set up the learning rate and decay.
    learning_rate = tf.train.inverse_time_decay(
      learning_rate=learning_rate,
      global_step=global_step,
      decay_steps=1,
      decay_rate=FLAGS.local_lr_decay,
      staircase=False)

    #  Set up Nesterov momentum optimizer.
    optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=FLAGS.momentum,
      use_nesterov=True)

    # Create training operation.
    train_op = slim.learning.create_train_op(loss, optimizer, global_step)

    # Add summaries.
    tf.summary.scalar('scores/train_loss', loss)
    tf.summary.scalar('dropout_keep_prob', FLAGS.dropout_keep_prob)
    tf.summary.scalar('learning_rate', learning_rate)

    # Summary writer.
    summary_writer = tf.summary.FileWriter(train_dir)

    # Saver.
    saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)
    best_model_saver = tf.train.Saver(max_to_keep=1)

    # Restore model from a checkpoint file if the path is given.
    restore_fn = None
    if previous_train_dir:
      def restore_model(sess):
        tf.logging.info('Restoring variables from the best checkpoint file {}'
                        .format(previous_train_dir))

        best_ckpt = tf.train.latest_checkpoint(
          checkpoint_dir=previous_train_dir,
          latest_filename=best_ckpt_latest_filename)
        saver.restore(sess, best_ckpt)

      restore_fn = restore_model

    def train_step(sess, train_op, global_step, train_step_kwargs):
      total_loss, should_stop = slim.learning.train_step(
        sess, train_op, global_step, train_step_kwargs)

      np_global_step = sess.run(global_step)

      if np_global_step % FLAGS.validation_every_n_steps == 0:
        mean_loss, roc_auc = evaluate(sess, val_pred, val_loss, val_labels,
                                      batch_size=FLAGS.batch_size,
                                      n_examples=FLAGS.n_val_examples)

        tf.logging.info('@ Validation scores: '
                        'mean_loss={:.4f}, roc_auc={:.4f}, global_step={}'
                        .format(mean_loss, roc_auc, np_global_step))

        summary = tf.Summary()
        summary.value.add(tag='scores/val_loss', simple_value=mean_loss)
        summary.value.add(tag='scores/val_roc_auc', simple_value=roc_auc)
        summary_writer.add_summary(summary, np_global_step)

        if mean_loss < train_step.best_loss:
          train_step.best_loss = mean_loss
          train_step.roc_auc_at_best_loss = roc_auc
          train_step.patience = FLAGS.patience

          tf.logging.info('@ A new best loss achieved! Saving the model...')

          best_model_saver.save(sess, best_ckpt_path,
                                global_step=global_step,
                                latest_filename=best_ckpt_latest_filename)
          _save_best_scores(train_dir, mean_loss, roc_auc)
        else:
          train_step.patience -= 1
          tf.logging.info('@ Validation loss did not decrease: '
                          'patience={}, step={}'
                          .format(train_step.patience, np_global_step))

          if train_step.patience < 1:
            should_stop = True
            tf.logging.info(
              '@ Patience is zero! Finish current training stage.')
            tf.logging.info('@ The validation scores of the best model were: '
                            'loss={}, roc_auc={}'
                            .format(train_step.best_loss,
                                    train_step.roc_auc_at_best_loss))

      return total_loss, should_stop

    # None if there is no previous scores
    best_loss, roc_auc_at_best_loss = _load_best_scores(train_dir)

    train_step.best_loss = best_loss or 100.0
    train_step.roc_auc_at_best_loss = roc_auc_at_best_loss or 0.0
    train_step.patience = FLAGS.patience

    # Kick off the training!
    slim.learning.train(
      train_op=train_op,
      train_step_fn=train_step,
      logdir=train_dir,
      graph=g,
      global_step=global_step,
      log_every_n_steps=FLAGS.log_every_n_steps,
      number_of_steps=FLAGS.max_steps,
      summary_writer=summary_writer,
      save_summaries_secs=60,
      save_interval_secs=600,
      init_fn=restore_fn,
      saver=saver)


def main(unused_argv):
  assert FLAGS.train_input_file_pattern, '--train_input_file_pattern is required'
  assert FLAGS.train_dir, '--train_dir is required'
  assert FLAGS.val_input_file_pattern, '--val_input_file_pattern is required'

  for i in range(FLAGS.max_trains):
    if os.path.isdir(_join_and_norm_path(FLAGS.train_dir, i + 1)):
      continue
      
    decay = FLAGS.global_lr_decay ** i
    learning_rate = FLAGS.initial_learning_rate * decay
    train_dir = _join_and_norm_path(FLAGS.train_dir, i)
    previous_train_dir = (_join_and_norm_path(FLAGS.train_dir, i - 1)
                          if i != 0 else None)

    print('''
    ###########################
    # Start training stage {} #
    ###########################'''.format(i))
    print('learning_rate={:.4f}'.format(learning_rate))
    print('train_dir={}'.format(train_dir))
    print()

    _train(learning_rate, train_dir, previous_train_dir)


if __name__ == '__main__':
  tf.app.run()
