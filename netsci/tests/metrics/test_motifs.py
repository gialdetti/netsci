import numpy as np
import numpy.testing as npt
from scipy.special import comb

import netsci.models.random as nsr
import netsci.metrics.motifs as nsm


def test_basic_patterns():
    print("Testing basic patterns..")
    M = nsm.triad_patterns()
    for i in np.arange(len(M)):
        motif = M[i].copy()
        print("Motif #%d\n%s" % (i, motif))
        f_naive = nsm.motifs(motif, algorithm='brute-force')
        npt.assert_equal(motif, M[i], "Argument array was altered during execution")
        print("\t%s %s" %(f_naive[0:3], f_naive[3:]))
        npt.assert_equal(np.where(f_naive == 1)[0], np.array([i]))

        if i >= 3:
            f = nsm.motifs(motif)
            npt.assert_equal(motif, M[i], "Argument array was altered during execution")
            print("\t%s %s" % (f[0:3], f[3:]))
            npt.assert_equal(np.where(f == 1)[0], np.array([i]))


def test_counts():
    n = 100
    p = 0.1

    A = nsr.erdos_renyi(n, p)

    print("Motifs (fast):")
    f = nsm.motifs(A)
    print("\tf = %s" % f)

    # print "Motifs (v0):"
    # f_v0 = motifs_v0(A)
    # print "\tf_v0 =", f_v0
    # print "Comparison: ", zip(f, f_v0)
    # npt.assert_equal(f[3:], f_v0[3:], "Motif frequencies of different versions are not compatible")

    print("Motifs (naive):")
    f_naive = nsm.motifs(A, algorithm='brute-force')
    print("\tf_naive = %s" % f_naive)
    print("Comparison: %s" % list(zip(f, f_naive)))
    npt.assert_equal(f[3:], f_naive[3:], "Motif frequencies from optimized algorithm are not correct")

    C3n = comb(n, 3)
    npt.assert_equal(sum(f_naive), C3n)


if False:
    import netsci.visualization.motifs as nsv

    n, p = 250, 0.2
    A = nsr.er(n, p)

    C3n = comb(n, 3)
    f_analytic = C3n * triads_mean(p)

    print("Counting motifs..")
    f = nsm.motifs(A)

    print("Plotting..")
    nsv.bar_motifs(f, f_analytic, triad_order=triad_order_nn4576, title="nn4576 triad order")
    nsv.bar_motifs(f, f_analytic, triad_order=triad_order_bct, title="BCT triad order")

