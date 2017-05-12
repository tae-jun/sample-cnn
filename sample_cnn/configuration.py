import tensorflow as tf


class ModelConfig:
  """Wrapper class for SampleCNN model hyperparameters."""

  def __init__(self, mode='train'):
    """Sets the default model hyperparameters."""
    assert mode in ['train', 'eval']

    self.mode = mode

    self.dropout_keep_prob = 0.5

    self.n_outputs = 50

    self.initializer = tf.contrib.layers.xavier_initializer
    self.activation_fn = tf.nn.relu
