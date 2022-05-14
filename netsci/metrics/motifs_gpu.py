import logging
import numpy as np


logger = logging.getLogger(__name__)
gpu = False

try:
    import tensorflow as tf
    logger.info('Imported tensorflow for MatMul/GPU support')
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
except ImportError:
    logger.debug('Could not import tensorflow for MatMul/GPU support')
    raise ImportError('Could not import tensorflow for MatMul/GPU support')


def to_gpu(A, dtype=tf.int64):
    # A = torch.from_numpy(A).int().to(device)
    A = tf.constant(A, dtype=dtype)
    return A


def _motifs_gpu(A, dtype=tf.float64):    
    A = to_gpu(A, dtype=dtype)

    A1 = tf.cast((A+tf.transpose(A)==1), dtype=dtype)*A
    A2 = tf.cast((A+tf.transpose(A)==2), dtype=dtype)
    assert tf.reduce_sum(A) == tf.reduce_sum(A1) + tf.reduce_sum(A2)

    A0 = tf.cast((A+tf.transpose(A)==0), dtype=dtype)
    A0 = tf.linalg.set_diag(A0, tf.zeros(len(A), dtype=dtype))
    assert tf.reduce_sum(A0) + tf.reduce_sum(A1)*2 + tf.reduce_sum(A2) == len(A)*(len(A) - 1)

    trace = tf.linalg.trace
    transpose = tf.transpose
    A1t = transpose(A1)

    logger.debug(f"Ax devices: {[x.device.split('/')[-1] for x in [A, A0, A1, A2]]}")

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
    logger.debug(f"f devices: {[x.device.split('/')[-1] for x in f]}")

    f = np.array(f, dtype=int)
    return f