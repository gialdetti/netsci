import time
import numpy as np
import numpy.testing as npt

from netsci.metrics.motifs import *

np.set_printoptions(linewidth=160)

n = 50; p = 0.1
# n = 100; p = 0.2
n = 500; p = 0.1

print(f"Generating {n}x{n} adjacency matrix (p={p:.4f})..")
A = np.random.binomial(1, p, size=(n, n))
np.fill_diagonal(A,0)

print("Motifs:")
start = time.time()
f = motifs(A)
end = time.time()
print(f"\tElapsed time {end - start:.4f} seconds")
print(f"\tf = {f}")

print("Motifs (naive):")
start = time.time()
f_naive = motifs_naive(A)
end = time.time()
print(f"\tElapsed time {end - start:.4f} seconds")
print(f"\tf_naive = {f_naive}")

print("Comparison: ", list(zip(f, f_naive)))
npt.assert_equal(f[3:], f_naive[3:], "Motif frequencies from optimized algorithm are not correct")


print("Motifs (with participation):")
start = time.time()
f_wp, participation = motifs_with_participation(A)
end = time.time()
print(f"\tElapsed time {end - start:4f} seconds")
print(f"\tf_wp = {f_wp}")
