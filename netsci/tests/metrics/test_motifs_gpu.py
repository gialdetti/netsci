import logging; logging.basicConfig()
logging.getLogger('netsci.metrics.motifs_gpu').setLevel(logging.DEBUG)

import numpy as np
import numpy.testing as npt

from netsci.models.random import erdos_renyi
import netsci.metrics.motifs as nsm


A = erdos_renyi(100, p=0.01)
f = nsm.motifs(A, algorithm='gpu')
print(f"f (matmul) = {f}")

f_expected = nsm.motifs(A)
print(f"f_expected = {f_expected}")
npt.assert_equal(f[3:], f_expected[3:])

f_expected_bf = nsm.motifs(A, algorithm='brute-force')
print(f"f_expected_bf = {f_expected_bf}")
npt.assert_equal(f, f_expected_bf)

