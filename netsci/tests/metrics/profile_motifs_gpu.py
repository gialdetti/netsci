import logging; logging.basicConfig()

import tensorflow as tf
logging.getLogger('netsci.metrics.motifs_gpu').setLevel(logging.DEBUG)

from netsci.models.random import erdos_renyi
import netsci.metrics.motifs as nsm
from netsci.tests.profiler import Timer

from tensorflow.python.client import device_lib


print('devices:', {d.name: d.physical_device_desc 
                   for d in device_lib.list_local_devices()})


A = erdos_renyi(n=2000, p=0.1, random_state=71070)
print(f'A: {A.shape}, p = {nsm.sparsity(A):.4f}')

configurations = [('matmul', 'GPU'), ('matmul', 'CPU'), ('louzoun', 'CPU')]

for algorithm, device in configurations:
  with tf.device(f'/{device}:0'):
    with Timer(logger=print, desc=f'{algorithm}:{device}'):
      print(nsm.motifs(A, algorithm=algorithm))


"""Colab
devices = {
  '/device:CPU:0': '', 
  '/device:GPU:0': 'device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7'
}

A: (2000, 2000), p = 0.1001

matmul:GPU took 0.680 sec.
matmul:CPU took 18.909 sec.
louzoun:CPU took 149.726 sec.
"""

"""MBP M1 Max
devices: {
  '/device:CPU:0': '', 
  '/device:GPU:0': 'device: 0, name: METAL, pci bus id: <undefined>'
}

A: (2000, 2000), p = 0.1001

matmul:GPU took 1.943 sec.
matmul:CPU took 1.868 sec.
louzoun:CPU took 64.622 sec.
"""
