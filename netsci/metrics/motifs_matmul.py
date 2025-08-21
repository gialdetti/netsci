import logging
from .tensorians import detect_tensorians

import numpy as np


logger = logging.getLogger(__name__)

tensorians = detect_tensorians()
priorities = ["torch-gpu", "tensorflow-gpu", "torch-cpu", "tensorflow-cpu", "numpy-cpu"]


default_tensorian = [t for t in priorities if t in tensorians.keys()][0]
logger.debug(f"'{default_tensorian}' tensorian was set as default.")


def motifs_matmul(A, tensorian=default_tensorian):
    logger.debug(f"tensorian: {tensorian}")
    tn = tensorians[tensorian]
    A = tn.to_gpu(A, dtype=tn.dtype)

    A1 = tn.cast(((A + tn.transpose(A)) == 1), dtype=tn.dtype) * A
    A2 = tn.cast(((A + tn.transpose(A)) == 2), dtype=tn.dtype)
    assert tn.reduce_sum(A) == tn.reduce_sum(A1) + tn.reduce_sum(A2)

    A0 = tn.cast(((A + tn.transpose(A)) == 0), dtype=tn.dtype)
    A0 = tn.set_diag(A0)
    assert tn.reduce_sum(A0) + tn.reduce_sum(A1) * 2 + tn.reduce_sum(A2) == len(A) * (
        len(A) - 1
    )

    A1t = tn.transpose(A1)

    logger.debug(
        f"Ax devices: {[str(x.device).split('/')[-1] for x in [A, A0, A1, A2]]}"
    )
    logger.debug(f"Ax dtypes: {[x.dtype for x in [A, A0, A1, A2]]}")

    # print("")
    # print(">>", tn.dtype, A, "||", A0, A1, A2, A1t)
    # print(">>", tn.dtype, A.dtype, "||", A0.dtype, A1.dtype, A2.dtype, A1t.dtype)
    # print(">>", tn.reduce_sum(A), "||", tn.reduce_sum(A0), tn.reduce_sum(A1))

    f = [
        # if dtype is int, than (?) should use floor division '//' instead of '/'. slower?
        tn.trace(A0 @ A0 @ A0) / 6,
        tn.trace(A1 @ A0 @ A0),
        tn.trace(A2 @ A0 @ A0) / 2,
        tn.trace(A1 @ A1 @ A0),
        tn.trace(A1 @ A1t @ A0) / 2,
        tn.trace(A1t @ A1 @ A0) / 2,
        tn.trace(A1 @ A1t @ A1),
        tn.trace(A2 @ A1 @ A0),
        tn.trace(A2 @ A1t @ A0),
        tn.trace(A1 @ A1 @ A1) / 3,
        tn.trace(A0 @ A2 @ A2) / 2,
        tn.trace(A1 @ A1 @ A2),
        tn.trace(A1 @ A1t @ A2) / 2,
        tn.trace(A1t @ A1 @ A2) / 2,
        tn.trace(A1 @ A2 @ A2),
        tn.trace(A2 @ A2 @ A2) / 6,
    ]
    logger.debug(f"f devices: {[str(x.device).split('/')[-1] for x in f]}")
    f = np.array([tn.to_cpu(x, dtype=None) for x in f]).astype(int)
    return f
