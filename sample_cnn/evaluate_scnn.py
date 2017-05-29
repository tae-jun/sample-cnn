import os

import tensorflow as tf

from sample_cnn.ops.inputs import batch_inputs
from sample_cnn.ops.evaluation import inference, evaluate

tf.flags.DEFINE_string('input_file_pattern', '',
                       'File pattern of sharded TFRecord input files.')
tf.flags.DEFINE_string('checkpoint_dir', '',
                       'Directory containing model checkpoints.')
tf.flags.DEFINE_bool('eval_best', True, 'Evaluate the best checkpoint.')
tf.flags.DEFINE_string('best_ckpt_name', 'best',
                       'Checkpoint name of the best model.')

tf.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.flags.DEFINE_integer('n_outputs', 50,
                        'Number of outputs (i.e. Number of tags).')
tf.flags.DEFINE_integer('num_examples', 4332, 'Number of examples to run.')

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS


def _eval_once():
  g = tf.Graph()
  with g.as_default():
    # sequence: [batch_size, 10, 58081]
    sequence, labels = batch_inputs(
      file_pattern=FLAGS.input_file_pattern,
      batch_size=FLAGS.batch_size,
      is_training=False,
      is_sequence=True,
      n_read_threads=1,
      examples_per_shard=120,
      shard_queue_name='filename_queue',
      example_queue_name='input_queue')

    # Validation.
    pred, loss = inference(sequence, labels)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)

    if FLAGS.eval_best:
      best_ckpt_latest_filename = FLAGS.best_ckpt_name + '_checkpoint'
      ckpt = tf.train.get_checkpoint_state(
        checkpoint_dir=FLAGS.checkpoint_dir,
        latest_filename=best_ckpt_latest_filename)
    else:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        save_path = ckpt.model_checkpoint_path
      else:
        save_path = os.path.join(FLAGS.checkpoint_dir,
                                 ckpt.model_checkpoint_path)

      tf.logging.info('Restoring variables from checkpoint file {}'
                      .format(save_path))
      saver.restore(sess, save_path)
    else:
      raise IOError('Could not find a checkpoint')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tf.logging.info('Start evaluation...')

    loss, roc_auc = evaluate(sess, pred, loss, labels,
                             n_examples=FLAGS.num_examples,
                             print_progress=True)

    tf.logging.info('loss={}, roc_auc={}'.format(loss, roc_auc))

    coord.request_stop()
    coord.join(threads)
    sess.close()


def main(unused_argv):
  assert FLAGS.input_file_pattern, '--input_file_pattern is required'
  assert FLAGS.checkpoint_dir, '--checkpoint_dir is required'

  _eval_once()


if __name__ == '__main__':
  tf.app.run()
