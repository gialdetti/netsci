import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Tensorian:
    library: str
    dtype: object
    to_gpu: Callable
    to_cpu: Callable
    cast: Callable
    transpose: Callable
    set_diag: Callable
    reduce_sum: Callable
    trace: Callable


def detect_tensorians():
    tensorians = {}

    try:
        import numpy as np

        def set_diag_np(tensor, value=0.0):
            np.fill_diagonal(tensor, 0)
            return tensor

        tensorians["numpy-cpu"] = Tensorian(
            library="numpy",
            # dtype=np.float32,
            dtype=np.float64,  # Slower, but less estimation errors using float32.
            # dtype=np.int64,
            to_gpu=lambda A, dtype: A.astype(dtype),
            to_cpu=lambda A, dtype: A,
            cast=lambda A, dtype: np.astype(A, dtype),
            transpose=np.transpose,
            set_diag=set_diag_np,
            reduce_sum=np.sum,
            trace=np.trace,
        )
    except ImportError:
        raise ImportError("Please install numpy")

    try:
        import tensorflow as tf

        def create_tf_tensorian(device=None, dtype=tf.float32):
            def set_diag_tf(tensor, value=0.0):
                tensor = tf.linalg.set_diag(
                    tensor, tf.zeros(len(tensor), dtype=tensor.dtype)
                )
                return tensor

            return Tensorian(
                library="tensorflow",
                dtype=dtype,
                to_gpu=lambda A, dtype: tf.constant(A, dtype=dtype),
                to_cpu=lambda T, dtype: T,
                cast=lambda A, dtype: tf.cast(A, dtype=dtype),
                transpose=tf.transpose,
                set_diag=set_diag_tf,
                reduce_sum=tf.reduce_sum,
                trace=tf.linalg.trace,
            )

        dtype = tf.float32

        tensorians["tensorflow-cpu"] = create_tf_tensorian(dtype=dtype)
        if len(tf.config.list_physical_devices("GPU")) > 0:
            print(f"Using GPU for tensorflow")
            device = tf.config.list_physical_devices("GPU")[0]
            tensorians["tensorflow-gpu"] = create_tf_tensorian(dtype=dtype)

    except ImportError:
        pass

    try:
        import torch

        def create_torch_tensorian(device, dtype):
            def set_diag_torch(tensor, value=0.0):
                tensor.fill_diagonal_(value)
                return tensor

            return Tensorian(
                library="torch",
                dtype=dtype,
                to_gpu=lambda A, dtype: torch.tensor(A, dtype=dtype, device=device),
                to_cpu=lambda T, dtype: T.cpu(),
                cast=lambda A, dtype: A.to(dtype=dtype),
                transpose=lambda A: torch.transpose(A, 0, 1),
                set_diag=set_diag_torch,
                reduce_sum=torch.sum,
                trace=torch.trace,
            )

        dtype = torch.float32

        tensorians["torch-cpu"] = create_torch_tensorian(torch.device("cpu"), dtype)
        if torch.cuda.is_available():
            print(
                f"Using CUDA-enabled GPU for pytorch: {torch.cuda.get_device_name(0)}"
            )
            tensorians["torch-gpu"] = create_torch_tensorian(
                torch.device("cuda"), dtype
            )
        elif torch.backends.mps.is_available():
            print("Using Apple Silicon GPU (MPS) for pytorch")
            # dtype = torch.float32
            # dtype = torch.int64  # Slower, but more accurate than float32?
            tensorians["torch-gpu"] = create_torch_tensorian(torch.device("mps"), dtype)

    except ImportError:
        pass

    return tensorians
