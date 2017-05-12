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
tf.flags.DEFINE_string('best_model_path', '',
                       'Directory where to save the best model.')
tf.flags.DEFINE_string('checkpoint_dir', '',
                       'Directory where to load model from checkpoints.')

# Batch options.
tf.flags.DEFINE_integer('batch_size', 23, 'Batch size.')
tf.flags.DEFINE_integer('n_train_examples', 187090,
                        'Number of examples in training dataset.')
tf.flags.DEFINE_integer('n_val_examples', 1825,
                        'Number of examples in validation dataset.')
tf.flags.DEFINE_integer('n_train_examples_per_shard', 1462,
                        'Approximate number of examples per training shard.')
tf.flags.DEFINE_integer('n_val_examples_per_shard', 152,
                        'Approximate number of examples per validation shard.')
tf.flags.DEFINE_integer('n_read_threads', 2, 'Number of example reader.')

# Learning options.
tf.flags.DEFINE_float('initial_learning_rate', 0.01, 'Initial learning rate.')
tf.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability.')
tf.flags.DEFINE_float('lr_decay', 0.2, 'Learning rate decay.')

# Training options.
tf.flags.DEFINE_integer('validation_every_n_steps', 8134,
                        'Number of steps to validate the model periodically.'
                        '(default 1 epoch)')
tf.flags.DEFINE_integer('patience', 3,
                        'If the validation loss does not decrease even after '
                        'learning `patience * validation_every_n_steps` more '
                        'steps, decay learning rate.')
tf.flags.DEFINE_integer('max_steps', 10000000, 'Number of batches to run.')
tf.flags.DEFINE_integer('log_every_n_steps', 5,
                        'Number of steps to log loss periodically.')
tf.flags.DEFINE_integer('max_checkpoints_to_keep', 1,
                        'Maximum number of checkpoints to keep.')

FLAGS = tf.app.flags.FLAGS


def _train():
  # Print all flags.
  for key, value in FLAGS.__flags.items():
    print(key, '=', value)

  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.logging.info('Creating training directory: %s', FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

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
      examples_per_shard=FLAGS.n_train_examples_per_shard,
      shard_queue_name='train_filename_queue',
      example_queue_name='train_input_queue')

    # sequence: [batch_size, 10, 58081]
    val_sequence, val_labels = batch_inputs(
      file_pattern=FLAGS.val_input_file_pattern,
      batch_size=FLAGS.batch_size,
      is_training=False,
      is_sequence=True,
      n_read_threads=1,
      examples_per_shard=FLAGS.n_val_examples_per_shard,
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
    n_lr_decay = tf.Variable(initial_value=0., trainable=False)
    decay_global_lr = n_lr_decay.assign_add(1.)
    lr = FLAGS.initial_learning_rate * tf.pow(FLAGS.lr_decay,
                                              n_lr_decay)

    #  Set up Nesterov momentum optimizer.
    optimizer = tf.train.MomentumOptimizer(
      learning_rate=lr,
      momentum=FLAGS.momentum,
      use_nesterov=True)

    # Create training operation.
    train_op = slim.learning.create_train_op(loss, optimizer, global_step)

    # Add summaries.
    tf.summary.scalar('losses/train_loss', loss)
    tf.summary.scalar('dropout_keep_prob', FLAGS.dropout_keep_prob)
    tf.summary.scalar('lr/lr', lr)
    tf.summary.scalar('lr/num_lr_decay', n_lr_decay)

    # Summary writer.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

    # Saver.
    saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)
    best_model_saver = tf.train.Saver(max_to_keep=1)

    # Restore model from a checkpoint file if the path is given.
    restore_fn = None
    if FLAGS.checkpoint_dir:
      def restore_model(sess):
        tf.logging.info('Restoring variables from checkpoint file {}'
                        .format(FLAGS.checkpoint_dir))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

      restore_fn = restore_model

    def train_step(sess, train_op, global_step, train_step_kwargs):
      total_loss, should_stop = slim.learning.train_step(
        sess, train_op, global_step, train_step_kwargs)

      np_global_step = sess.run(global_step)

      if np_global_step % FLAGS.validation_every_n_steps == 0:
        mean_loss, roc_auc = evaluate(sess, val_pred, val_loss, val_labels,
                                      n_examples=FLAGS.n_val_examples)

        tf.logging.info('Validation scores: '
                        'mean_loss={:.4f}, roc_auc={:.4f}, global_step={}'
                        .format(mean_loss, roc_auc, np_global_step))

        summary = tf.Summary()
        summary.value.add(tag='validation/loss', simple_value=mean_loss)
        summary.value.add(tag='validation/roc_auc', simple_value=roc_auc)
        summary_writer.add_summary(summary, np_global_step)

        if mean_loss < train_step.best_loss:
          train_step.best_loss = mean_loss
          train_step.patience = FLAGS.patience

          tf.logging.info('A new best loss achieved! Saving the model...')

          best_model_saver.save(sess, FLAGS.best_model_path,
                                global_step=global_step,
                                latest_filename='best_checkpoint')
        else:
          train_step.patience -= 1
          tf.logging.info('Validation loss did not decrease: '
                          'patience={}, step={}'
                          .format(train_step.patience, np_global_step))

          if train_step.patience < 1:
            np_lr, _ = sess.run([lr, decay_global_lr])

            tf.logging.info('Patience is zero! Decay learning rate to {:.2e}: '
                            'global_step={}'.format(np_lr, np_global_step))

            train_step.patience = FLAGS.patience

      return total_loss, should_stop

    train_step.best_loss = 100.0
    train_step.patience = FLAGS.patience

    # Kick off the training!
    slim.learning.train(
      train_op=train_op,
      train_step_fn=train_step,
      logdir=FLAGS.train_dir,
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
  assert FLAGS.best_model_path, '--best_model_path is required'
  assert FLAGS.val_input_file_pattern, '--val_input_file_pattern is required'

  _train()


if __name__ == '__main__':
  tf.app.run()
