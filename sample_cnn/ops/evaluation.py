import tensorflow as tf
import numpy as np

from keras.layers import Input
from sklearn.metrics import roc_auc_score, log_loss

from sample_cnn.ops import batch_inputs
from sample_cnn.model import SampleCNN


def evaluate(input_file_pattern, weights_path, n_examples,
             n_audios_per_shard=100, print_progress=True):
  # audio: [1, 10, 58081]
  # labels: [1, 50]
  audio, labels = batch_inputs(
    file_pattern=input_file_pattern,
    batch_size=1,
    is_training=False,
    is_sequence=True,
    n_read_threads=1,
    examples_per_shard=n_audios_per_shard,
    shard_queue_name='filename_queue',
    example_queue_name='input_queue')

  segments = tf.squeeze(audio)

  labels = tf.squeeze(labels)
  labels = Input(tensor=labels)

  model = SampleCNN(segments=segments,
                    extra_inputs=labels,
                    extra_outputs=labels)

  print('Load weights from "{}".'.format(weights_path))
  model.load_weights(weights_path)

  n_classes = labels.shape[0].value
  all_y_pred = np.empty([0, n_classes], dtype=np.float32)
  all_y_true = np.empty([0, n_classes], dtype=np.float32)

  print('Start evaluation.')
  for i in range(n_examples):
    y_pred_segments, y_true = model.predict_tfrecord(segments)

    y_pred = np.mean(y_pred_segments, axis=0)

    y_pred = np.expand_dims(y_pred, 0)
    y_true = np.expand_dims(y_true, 0)

    all_y_pred = np.append(all_y_pred, y_pred, axis=0)
    all_y_true = np.append(all_y_true, y_true, axis=0)

    if print_progress and i % (n_examples // 100) == 0 and i:
      print('Evaluated [{:04d}/{:04d}].'.format(i + 1, n_examples))

  losses = []
  for i in range(n_classes):
    class_y_true = all_y_true[:, i]
    class_y_pred = all_y_pred[:, i]
    if np.sum(class_y_true) != 0:
      class_loss = log_loss(class_y_true, class_y_pred)
      losses.append(class_loss)

  loss = np.mean(losses)
  print('@ binary cross entropy loss: {}'.format(loss))

  roc_auc = roc_auc_score(all_y_true, all_y_pred, average='macro')
  print('@ ROC AUC score: {}'.format(roc_auc))
