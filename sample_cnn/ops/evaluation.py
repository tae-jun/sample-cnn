from math import ceil

import tensorflow as tf
import numpy as np

from sklearn.metrics import roc_auc_score

from sample_cnn.model.sample_cnn import SampleCNN
from sample_cnn.configuration import ModelConfig


def _safe_auc_roc(label, pred):
  label_col_sum = label.sum(axis=0)
  zero_label_col = label_col_sum == 0
  n_zero_label = zero_label_col.sum()

  if n_zero_label > 0:
    tf.logging.warn('Only zeros exist in {} class(es). '
                    'The ROC AUC will be calculated excluding the class(es).'
                    .format(n_zero_label))

    non_zero_label = label[:, ~zero_label_col]
    non_zero_pred = pred[:, ~zero_label_col]

    return roc_auc_score(non_zero_label, non_zero_pred, average='macro')
  else:
    return roc_auc_score(label, pred, average='macro')


def inference(sequence, labels, reuse=True):
  # Build model for evaluation.
  config = ModelConfig(mode='eval')
  scnn = SampleCNN(config, reuse=reuse)

  _, n_segments, _ = sequence.shape

  # Get run-time batch size since the last batch size is smaller than others.
  batch_size = tf.shape(sequence)[0]

  # Tensor of shape [10, batch, 58081]
  segments = tf.transpose(sequence, [1, 0, 2])

  def _inference(i, pred_cumsum, loss_cumsum):
    logits = scnn(segments[i])

    seg_loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    seg_pred = tf.nn.sigmoid(logits)

    return (i + 1,
            tf.add(pred_cumsum, seg_pred),
            tf.add(loss_cumsum, seg_loss))

  _, pred_sum, loss_sum = tf.while_loop(
    cond=lambda i, *_: i < n_segments,
    body=_inference,
    loop_vars=(
      tf.constant(0),
      tf.zeros([batch_size, config.n_outputs]),
      tf.zeros([batch_size])),
    back_prop=False)

  pred = tf.div(pred_sum, tf.cast(n_segments, dtype=tf.float32))

  seg_loss_mean = tf.div(loss_sum, tf.cast(n_segments, dtype=tf.float32))
  loss = tf.reduce_mean(seg_loss_mean)

  return pred, loss


def evaluate(sess, pred, loss, labels, batch_size, n_examples,
             print_progress=False):
  _, n_outputs = labels.shape
  num_iter = int(ceil(1.0 * n_examples / batch_size))

  total_loss = 0.0
  total_preds = np.empty([0, n_outputs], dtype=np.float32)
  total_labels = np.empty([0, n_outputs], dtype=np.float32)

  for i in range(num_iter):
    if print_progress:
      tf.logging.info('[{}/{}] Evaluating...'.format(i + 1, num_iter))

    batch_pred, batch_loss, batch_labels = sess.run([pred, loss, labels])

    total_loss += batch_loss
    total_preds = np.append(total_preds, batch_pred, axis=0)
    total_labels = np.append(total_labels, batch_labels, axis=0)

  mean_loss = total_loss / num_iter
  roc_auc = _safe_auc_roc(total_labels, total_preds)

  return mean_loss, roc_auc
