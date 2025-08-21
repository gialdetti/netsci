import logging

logging.basicConfig()
logging.getLogger("netsci").setLevel(logging.DEBUG)

import numpy as np
import numpy.testing as npt

from netsci.models.random import erdos_renyi
import netsci.metrics.motifs as nsm


def test_motifs():
    A = erdos_renyi(100, p=0.01)
    f = nsm.motifs(A, algorithm="matmul")
    print(f"f (matmul) = {f}")

    f_expected = nsm.motifs(A)
    print(f"f_expected = {f_expected}")
    npt.assert_equal(f[3:], f_expected[3:])

    f_expected_bf = nsm.motifs(A, algorithm="brute-force")
    print(f"f_expected_bf = {f_expected_bf}")
    npt.assert_equal(f, f_expected_bf)


def has_torch(require_gpu=None):
    try:
        import torch

        has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        if require_gpu:
            return has_gpu
        else:
            return True
    except ImportError:
        return False


def has_tensorflow(require_gpu=None):
    try:
        import tensorflow as tf

        has_gpu = len(tf.config.list_physical_devices("GPU")) > 0
        if require_gpu:
            return has_gpu
        else:
            return True
    except ImportError:
        return False


def test_libraries():
    tensorians = nsm.motifs_matmul.tensorians.keys()

    assert "numpy-cpu" in tensorians

    if has_torch():
        assert "torch-cpu" in tensorians
        if has_torch(require_gpu=True):
            assert "torch-gpu" in tensorians

    if has_tensorflow():
        assert "tensorflow-cpu" in tensorians
        if has_tensorflow(require_gpu=True):
            assert "tensorflow-gpu" in tensorians


def test_default_matmul():
    default_tensorian = nsm.motifs_matmul.default_tensorian
    if has_torch():
        assert "torch-" in default_tensorian
        assert (
            "torch-gpu" if has_torch(require_gpu=True) else "torch-cpu"
        ) in default_tensorian
    elif has_tensorflow():
        assert "tensorflow-" in default_tensorian
        assert (
            "tensorflow-gpu" if has_tensorflow(require_gpu=True) else "tensorflow-cpu"
        ) in default_tensorian
    else:
        assert "numpy-cpu" in default_tensorian
