import itertools

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

import netsci.metrics.motifs as nsm


def permute_array(X, order):
    order = np.array(order)
    return np.array([row[order] for row in X[order]])


def test_a_single_pattern_permuations():
    tid = 6
    M = nsm.triad_patterns()[tid]
    # plot_a_triad(M, r=.5)

    A = np.zeros(shape=(5, 5), dtype=np.int)
    A[1:4, 1:4] = M

    expected_f = np.eye(16, dtype=np.int)[tid]
    for permutation in itertools.permutations(range(5), 5):
        print("# permutation:", permutation)
        perm_B = permute_array(A, permutation)
        print(perm_B)

        f = nsm.motifs(perm_B)
        f[0:3] = 0
        npt.assert_equal(f, expected_f)

        f, participation = nsm.motifs(perm_B, participation=True)
        f[0:3] = 0
        print(participation)
        npt.assert_equal(f, expected_f)
        npt.assert_equal([len(p) for p in participation][3:], f[3:], "Frequency and participation lists are not consistent")

        r, g, b = [np.where(np.array(permutation)==i)[0][0] for i in [1,2,3]]
        print("(R=%d, G=%d, B=%d) vs. participation=%s\n" % (r, g, b, participation[tid][0]))
        npt.assert_array_equal(participation[tid][0], [r, g, b])


def test_geometrical_network():
    n = 10
    Ys = np.random.rand(n)
    Is, Js = np.mgrid[0:n, 0:n]
    P = (Ys[Is] < Ys[Js]).astype(np.int8)

    A = np.random.binomial(1, P)
    f, participation = nsm.motifs(A, participation=True)
    npt.assert_equal([len(p) for p in participation][3:], f[3:], "Frequency and participation lists are not consistent")

    part6 = participation[6]
    assert np.all([(Ys[r] < Ys[g] < Ys[b]) for (r,g,b) in part6])

    fy = Ys[np.array(part6)]
    plt.plot([1,2,3], fy.T,'-o', zorder=1, alpha=.25, lw=1)
    plt.errorbar([1,2,3], fy.mean(axis=0), fy.std(axis=0),
                 lw=2, color='k', zorder=2)