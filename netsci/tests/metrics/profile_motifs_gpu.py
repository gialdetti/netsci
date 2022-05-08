import logging; logging.basicConfig()
import numpy as np
import numpy.testing as npt

import tensorflow as tf
logging.getLogger('netsci.metrics.motifs_gpu').setLevel(logging.DEBUG)

from netsci.models.random import erdos_renyi
import netsci.metrics.motifs as nsm
from netsci.metrics.motifs_gpu import _motifs_gpu

from netsci.tests.profiler import Timer



A = erdos_renyi(n=1000, p=0.1, random_state=71070)


# with Timer(logger=print, desc='GPU') as t:
#   with tf.device('/GPU:0'):
#     print(nsm.motifs(A, algorithm='gpu'))

# with Timer(logger=print, desc='CPU'):
#   with tf.device('/CPU:0'):
#     print(nsm.motifs(A, algorithm='gpu'))
with Timer(logger=print, desc='GPU') as t:
  with tf.device('/GPU:0'):
    print(_motifs_gpu(A, dtype=tf.int64))

with Timer(logger=print, desc='CPU'):
  with tf.device('/CPU:0'):
    print(_motifs_gpu(A, dtype=tf.int64))


with Timer(logger=print):
    print(nsm.motifs(A))


# with Timer(logger=print, desc='brute-force'):
#     print(nsm.motifs(A, algorithm='brute-force'))


import pandas as pd
times = pd.read_csv('/Users/eyalgal/Downloads/gpu-speedup-times.Tesla-K80-Colab.(220508.122555).csv',
 parse_dates=['timestamp'])

import seaborn as sns
sns.relplot(x='n', y='computation_time', hue='p', col='device', data=times, 
            kind='line', markers=True, style='device', dashes=False)


speedup = times.groupby(['n', 'p', 'device', 'random_state'])['computation_time'].mean().unstack(['device']) \
  .assign(speedup=lambda df: df['/CPU:0']/df['/GPU:0']).reset_index()

sns.relplot(x='n', y='speedup', hue='p', data=speedup, 
            kind='line', markers=True)

    