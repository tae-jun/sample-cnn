import tensorflow as tf

from sample_cnn.ops import evaluate

tf.flags.DEFINE_string('input_file_pattern', '',
                       'File pattern of sharded TFRecord input files.')
tf.flags.DEFINE_string('weights_path', '', 'Path to learned weights.')

tf.flags.DEFINE_integer('n_examples', 4332, 'Number of examples to run.')
tf.flags.DEFINE_integer('n_audios_per_shard', 100,
                        'Number of audios per shard.')

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS


def main(unused_argv):
  assert FLAGS.input_file_pattern, '--input_file_pattern is required'
  assert FLAGS.weights_path, '--weights_path is required'

  evaluate(FLAGS.input_file_pattern, FLAGS.weights_path,
           n_examples=FLAGS.n_examples,
           n_audios_per_shard=FLAGS.n_audios_per_shard)


if __name__ == '__main__':
  tf.app.run()
