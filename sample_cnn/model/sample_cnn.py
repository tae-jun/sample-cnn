import tensorflow as tf
import tensorflow.contrib.slim as slim


class SampleCNN:
  """Represents Sample CNN."""
  def __init__(self, config):
    self.config = config
    self.scope = 'SampleCNN'

  def __call__(self, input):
    outputs = self._build_net(input)
    logits = self._build_logits(outputs)
    return logits

  def outputs(self, input):
    outputs = self._build_net(input)
    return outputs

  @property
  def is_training(self):
    return self.config.mode == 'train'

  @property
  def trainable(self):
    return self.is_training

  @property
  def batch_norm_params(self):
    return {
      'is_training': self.is_training,
      'trainable': self.trainable
    }

  def _build_net(self, input):
    with tf.variable_scope(self.scope, reuse=True):
      with slim.arg_scope(
              [slim.convolution],
              stride=1,
              padding='SAME',
              weights_initializer=self.config.initializer(),
              trainable=self.trainable,
              activation_fn=self.config.activation_fn,
              normalizer_fn=slim.batch_norm,
              normalizer_params=self.batch_norm_params):
        with slim.arg_scope(
                [slim.pool],
                stride=3,
                padding='VALID',
                pooling_type='MAX'):
          conv1d = slim.convolution
          max_pool1d = slim.pool

          # 59049
          net = tf.expand_dims(input, -1)
          # 59049 X 1
          net = conv1d(net, 128, 3, stride=3, scope='Conv1d_0')
          # 19683 X 128
          net = conv1d(net, 128, 3, scope='Conv1d_1')
          net = max_pool1d(net, 3, scope='MaxPool1d_1')
          # 6561 X 128
          net = conv1d(net, 128, 3, scope='Conv1d_2')
          net = max_pool1d(net, 3, scope='MaxPool1d_2')
          # 2187 X 128
          net = conv1d(net, 256, 3, scope='Conv1d_3')
          net = max_pool1d(net, 3, scope='MaxPool1d_3')
          # 729 X 256
          net = conv1d(net, 256, 3, scope='Conv1d_4')
          net = max_pool1d(net, 3, scope='MaxPool1d_4')
          # 243 X 256
          net = conv1d(net, 256, 3, scope='Conv1d_5')
          net = max_pool1d(net, 3, scope='MaxPool1d_5')
          # 81 X 256
          net = conv1d(net, 256, 3, scope='Conv1d_6')
          net = max_pool1d(net, 3, scope='MaxPool1d_6')
          # 27 X 256
          net = conv1d(net, 256, 3, scope='Conv1d_7')
          net = max_pool1d(net, 3, scope='MaxPool1d_7')
          # 9 X 256
          net = conv1d(net, 256, 3, scope='Conv1d_8')
          net = max_pool1d(net, 3, scope='MaxPool1d_8')
          # 3 X 256
          net = conv1d(net, 512, 3, scope='Conv1d_9')
          net = max_pool1d(net, 3, scope='MaxPool1d_9')
          # 1 X 512
          net = conv1d(net, 512, 1, stride=1, scope='Conv1d_10')
          # 1 X 512

          net = slim.dropout(net,
                             keep_prob=self.config.dropout_keep_prob,
                             is_training=self.is_training,
                             scope='Dropout')
          net = slim.flatten(net, scope='Flatten')

          return net

  def _build_logits(self, outputs):
    logits = slim.fully_connected(
      outputs,
      num_outputs=self.config.n_outputs,
      activation_fn=None,
      weights_initializer=self.config.initializer(),
      scope='Logits',
      reuse=True)

    return logits
