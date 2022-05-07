import logging
import numpy as np


logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    logger.info('Imported tensorflow for GPU support')
except ImportError:
    logger.debug('Could not import tensorflow for GPU support')

    try:
        import torch
    except ImportError:
        logger.debug('Could not import torch for GPU support')

        raise ImportError('Could not initiate GPU support')


def to_gpu(A):
    # A = torch.from_numpy(A).int().to(device)
    A = tf.constant(A, dtype=tf.int64)
    return A


def _motifs_gpu(A):
    A = to_gpu(A)

    A1 = tf.cast((A+tf.transpose(A)==1), tf.int64)*A
    A2 = tf.cast((A+tf.transpose(A)==2), tf.int64)
    assert tf.reduce_sum(A) == tf.reduce_sum(A1) + tf.reduce_sum(A2)

    A0 = tf.cast((A+tf.transpose(A)==0), tf.int64)
    A0 = tf.linalg.set_diag(A0, tf.zeros(len(A), dtype=tf.int64))
    assert tf.reduce_sum(A0) + tf.reduce_sum(A1)*2 + tf.reduce_sum(A2) == len(A)*(len(A) - 1)

    trace = tf.linalg.trace
    transpose = tf.transpose
    A1t = transpose(A1)

    # logger.debug(A.device, A0.device, A1.device, A2.device)

    f = [
        trace(A0 @ A0 @ A0) / 6,
        
        trace(A1 @ A0 @ A0),

        trace(A2 @ A0 @ A0) / 2,
        trace(A1 @ A1 @ A0),
        trace(A1 @ A1t @ A0) / 2,
        trace(A1t @ A1 @ A0) / 2,

        trace(A1 @ A1t @ A1),
        trace(A2 @ A1 @ A0),
        trace(A2 @ A1t @ A0),
        trace(A1 @ A1 @ A1) / 3,

        trace(A0 @ A2 @ A2) / 2,

        trace(A1 @ A1 @ A2),
        trace(A1 @ A1t @ A2) / 2,
        trace(A1t @ A1 @ A2) / 2,

        trace(A1 @ A2 @ A2),
        
        trace(A2 @ A2 @ A2) / 6
    ]

    f = np.array(f, dtype=int)
    return f