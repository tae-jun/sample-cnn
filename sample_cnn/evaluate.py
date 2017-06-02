import numpy as np
import tensorflow as tf
from keras.layers import Input
from sklearn.metrics import roc_auc_score, log_loss

from sample_cnn.inputs import batch_inputs
from sample_cnn.model import SampleCNN

tf.flags.DEFINE_string('input_file_pattern', '',
                       'File pattern of sharded TFRecord input files.')
tf.flags.DEFINE_string('weights_path', '', 'Path to learned weights.')

tf.flags.DEFINE_integer('n_examples', 4332, 'Number of examples to run.')
tf.flags.DEFINE_integer('n_audios_per_shard', 100,
                        'Number of audios per shard.')

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS


def eval_once():
  g = tf.Graph()
  with g.as_default():
    # sequence: [1, 10, 58081]
    # labels: [1, 50]
    sequence, labels = batch_inputs(
      file_pattern=FLAGS.input_file_pattern,
      batch_size=1,
      is_training=False,
      is_sequence=True,
      n_read_threads=1,
      examples_per_shard=FLAGS.n_audios_per_shard,
      shard_queue_name='filename_queue',
      example_queue_name='input_queue')

    segments = tf.squeeze(sequence)

    labels = tf.squeeze(labels)
    labels = Input(tensor=labels)

    model = SampleCNN(segments=segments,
                      extra_inputs=labels,
                      extra_outputs=labels)

    print('Load weights from "{}".'.format(FLAGS.weights_path))
    model.load_weights(FLAGS.weights_path)

    n_classes = labels.shape[0].value
    all_y_pred = np.empty([0, n_classes], dtype=np.float32)
    all_y_true = np.empty([0, n_classes], dtype=np.float32)

    print('Start evaluation.')
    for i in range(FLAGS.n_examples):
      y_pred_segments, y_true = model.predict_tfrecord(segments)

      y_pred = np.mean(y_pred_segments, axis=0)

      y_pred = np.expand_dims(y_pred, 0)
      y_true = np.expand_dims(y_true, 0)

      all_y_pred = np.append(all_y_pred, y_pred, axis=0)
      all_y_true = np.append(all_y_true, y_true, axis=0)

      if i % (FLAGS.n_examples // 100) == 0 and i:
        print('Evaluated [{:04d}/{:04d}].'.format(i + 1, FLAGS.n_examples))

    losses = []
    roc_aucs = []
    for i in range(n_classes):
      class_y_true = all_y_true[:, i]
      class_y_pred = all_y_pred[:, i]

      if np.sum(class_y_true) != 0:
        class_loss = log_loss(class_y_true, class_y_pred)
        class_roc_auc = roc_auc_score(class_y_true, class_y_pred)

        losses.append(class_loss)
        roc_aucs.append(class_roc_auc)

    loss = np.mean(losses)
    roc_auc = np.mean(roc_aucs)
    print('@ binary cross entropy loss: {}'.format(loss))
    print('@ ROC AUC score: {}'.format(roc_auc))

    roc_auc = roc_auc_score(all_y_true, all_y_pred, average='macro')
    print('@ scikit-learn ROC AUC score: {}'.format(roc_auc))


def main(unused_argv):
  assert FLAGS.input_file_pattern, '--input_file_pattern is required'
  assert FLAGS.weights_path, '--weights_path is required'

  eval_once()


if __name__ == '__main__':
  tf.app.run()
