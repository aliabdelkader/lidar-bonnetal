# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import tensorflow as tf
import numpy as np
import scipy.misc
try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x

from tensorboard.plugins.hparams import api as hp

class Logger(object):

  def __init__(self, log_dir, hparams: list, metrics: list):
    """Create a summary writer logging to log_dir."""
    self.writer = tf.summary.create_file_writer(log_dir)
    if (hparams is not None) and (metrics is not None):
      with self.writer.as_default():
        hp.hparams_config(hparams=hparams, metrics=metrics)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    with self.writer.as_default():
      tf.summary.scalar(tag, value, step=step)
      self.writer.flush()

  def image_summary(self, tag, images, step):
    """Log a list of images."""
    images_array = np.stack(images)
    with self.writer.as_default():
      tf.summary.image(tag, images_array, step=step)
  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""
    with self.writer.as_default():
      tf.summary.histogram(tag, values, step=step, buckets=bins)

  def log_hyper_parameters(self, hparams: dict):
     with self.writer.as_default():
       hp.hparams(hparams)

