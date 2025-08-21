import logging

from netsci.system_info import get_system_info
from netsci.models.random import erdos_renyi
import netsci.metrics.motifs as nsm
from netsci.tests.profiler import Timer

logging.basicConfig()
# logging.getLogger("netsci.metrics.motifs_gpu").setLevel(logging.DEBUG)


print(get_system_info()["gpus_compute"])


A = erdos_renyi(n=2000, p=0.1, random_state=71070)
print(f"A: {A.shape}, p = {nsm.sparsity(A):.4f}")

for algorithm in ["louzoun"]:
    with Timer(logger=print, desc=f"{algorithm}"):
        print(nsm.motifs(A, algorithm=algorithm))

for tensorian, _ in nsm.motifs_matmul.tensorians.items():
    with Timer(logger=print, desc=f"{tensorian}"):
        print(nsm.motifs_matmul.motifs_matmul(A, tensorian))


"""Tesla T4 @ Colab
A: (2000, 2000), p = 0.1001

louzoun took 127.716 sec.
numpy-cpu took 9.627 sec.
tensorflow-cpu took 0.626 sec.
tensorflow-gpu took 0.129 sec.
torch-cpu took 4.698 sec.
torch-gpu took 0.390 sec.
"""

"""MBP M3 Max
A: (2000, 2000), p = 0.1001

louzoun took 47.796 sec.
numpy-cpu took 0.690 sec.
torch-cpu took 0.209 sec.
torch-gpu took 0.168 sec.
"""
