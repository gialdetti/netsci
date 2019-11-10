__author__ = 'gialdetti'

import itertools
from operator import itemgetter

import numpy as np
import networkx as nx


def sparsity(A):
    """The marginal probability for a connection

    Parameters
    ----------
    A : 2d-array
        Binary adjacency matrix.

    Returns
    -------
    p : float
        The marginal probability for a connection (network sparsity).

    See Also
    --------
    reciprocity

    """
    assert len(A.shape)==2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    return A.sum() / float(n*(n-1))


def reciprocity(A):
    """The marginal probability for a reciprocal connection

    Parameters
    ----------
    A : 2d-array
        Binary adjacency matrix.

    Returns
    -------
    r : float
        The marginal probability for a reciprocal connection.

    See Also
    --------
    sparsity

    """
    assert len(A.shape)==2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    return (A+A.T == 2).sum() / float(n*(n-1))


def motifs(A, algorithm='louzoun', participation=False):
    """Frequency of triplet motifs

    Parameters
    ----------
    A : 2d-array
        Binary adjacency matrix, where `A[i,j]==1` denotes an existing connection from node `i` to node `j`.
    algorithm : str, {'louzoun', 'brute-force'}, default: 'louzoun',
        Algorithm to use in the optimization problem.
        * 'brute-force' - the naive implementation using brute force algorithm. The complexity is high (O(|V|^3)), but it
          but it counts all 16 triplets.
        * 'louzoun' - an efficient algorithm for sparse networks. The complexity is low (O(|E|)), but it counts only the 13
          connected triplets (the first 3 entries will be -1).
    participation : bool, default: False
        If True, then all unique instances of motifs will be kept during analysis.
        Otherwise (default), this step will be avoided and list will not be returned.
        Note that collecting all instances may strongly increase the process time of the analysis.


    Returns
    -------
    f : array
        The frequencies of all 16 triplet motifs.
    participants : list of lists, optional
        A list of all instances for each motif. Only returned when `participation=True`.

    Examples
    --------
    Analyzing a star network (of four nodes)
    >>> A = np.array([[0,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]])
    >>> motifs(A)
    array([1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    References
    --------
    * Itzhack, R., Mogilevski, Y., and Louzoun, Y. (2007). An optimal algorithm for counting network motifs.
      Phys. A Stat. Mech. Its Appl. 381, 482-490.
    * Milo, R., Shen-Orr, S., ..., and Alon, U. (2002). Network motifs: Simple building blocks of complex networks.
      Science (80). 298, 824-827.

    """

    impls = {
        False: {'louzoun': _motifs, 'brute-force': _motifs_naive},
        True: {'louzoun' : _motifs_with_participation}
    }
    try:
        return impls[participation][algorithm](A)
    except KeyError as error:
        if not type(participation)==bool:
            raise TypeError(f'participation must boolean')
        raise ValueError(f'algorithm must be from {list(impls[participation].keys())}, got \'{algorithm}\'')


def _motifs_naive(A):
    A = A+2*A.T
    tags = triad_classification_tree()

    f = np.zeros(16, dtype=int)
    n = A.shape[1]
    for i in range(0, n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tag = tags[A[i,j], A[i,k], A[j,k]]
                f[ tag ] += 1

    return f


def _motifs(A):
    assert not A.diagonal().any(), "Diagonal should be all zeros"
    assert issubclass(A.dtype.type, np.integer), "Adjacency matrix should be a zero/one array"
    tags = triad_classification_tree()
    f = np.zeros(16, dtype=int)

    g = nx.Graph(A)
    A = A + 2*A.T
    for u in list(g.nodes()):
        u_nbrs = list(g.neighbors(u))
        u_nbrs_set = set(u_nbrs)
        g.remove_node(u)

        for j in range(len(u_nbrs)):
            v = u_nbrs[j]
            Auv = A[u,v]

            for k in range(j+1, len(u_nbrs)):
                w = u_nbrs[k]
                f[tags[Auv, A[u, w], A[v, w]]] += 1

            v_nbrs = set(g.neighbors(v)) - u_nbrs_set
            for w in v_nbrs:
                f[tags[Auv, A[u, w], A[v, w]]] += 1

    f[0:3] = -1
    return f


def _motifs_with_participation(A):
    assert not A.diagonal().any(), "Diagonal should be all zeros."
    tags = triad_classification_tree_with_participation()

    f = np.zeros(16, dtype=int)
    participants = l = [[] for _ in range(16)]

    g = nx.Graph(A)
    A = A + 2*A.T
    for u in list(g.nodes()):
        u_nbrs = list(g.neighbors(u))
        u_nbrs_set = set(u_nbrs)
        g.remove_node(u)

        for j in range(len(u_nbrs)):
            v = u_nbrs[j]
            Auv = A[u,v]

            for k in range(j+1, len(u_nbrs)):
                w = u_nbrs[k]
                tag = tags[Auv, A[u, w], A[v, w]]
                f[tag[0]] += 1
                participants[tag[0]].append(itemgetter(*tag[1:])([u,v,w]))

            v_nbrs = set(g.neighbors(v)) - u_nbrs_set
            for w in v_nbrs:
                tag = tags[Auv, A[u, w], A[v, w]]
                f[tag[0]] += 1
                participants[tag[0]].append(itemgetter(*tag[1:])([u,v,w]))

    f[0:3] = -1
    participants = [np.array(p, np.int).tolist() for p in participants]
    return f, participants


def triad_patterns():
    """Patterns of all 16 triads

    Returns
    -------
    patterns : list of 3x3 arrays
        The adjacency matrices of all 16 triads.


    """
    patterns = [
        # No edge
        np.array([[0,0,0], [0,0,0], [0,0,0]]),      #0
        # One edge
        np.array([[0,1,0], [0,0,0], [0,0,0]]),      #1
        np.array([[0,1,0], [1,0,0], [0,0,0]]),      #2
        # Two edges
        np.array([[0,1,0], [0,0,0], [1,0,0]]),      #4
        np.array([[0,0,0], [1,0,0], [1,0,0]]),      #3
        np.array([[0,1,1], [0,0,0], [0,0,0]]),      #5
        # Three edges
        np.array([[0,1,1], [0,0,1], [0,0,0]]),      #7
        np.array([[0,0,0], [0,0,1], [1,1,0]]),      #8
        np.array([[0,1,0], [0,0,1], [0,1,0]]),      #6
        np.array([[0,1,0], [0,0,1], [1,0,0]]),      #9
        # Four edges
        np.array([[0,1,1], [1,0,0], [1,0,0]]),      #11
        np.array([[0,1,0], [0,0,1], [1,1,0]]),      #12
        np.array([[0,0,0], [1,0,1], [1,1,0]]),      #13
        np.array([[0,1,1], [0,0,1], [0,1,0]]),      #10
        # Five edges
        np.array([[0,1,1], [1,0,1], [1,0,0]]),      #14
        # Six edges
        np.array([[0,1,1], [1,0,1], [1,1,0]])       #15
    ]
    return patterns


def triad_proba(p, r=None):
    """The triad distribution in an Erdős–Rényi-like network

    For some graph models, the probability for finding a particular triad pattern among three randomly chosen nodes
    can be computed analytically. Specifically, the Erdős–Rényi model and it's reciprocal extension are such tractable
    models. The function computes probability for each of the 16 triads.

    Parameters
    ----------
    p : float
        The marginal probability for a connection (network sparsity).
    r : float, optional
        The marginal probability for a reciprocal connection.

    Returns
    -------
    mu : array
        The probability for each triad

    """
    if r is None:
        r = p**2

    alpha = r-p**2
    z = np.sqrt(1-alpha)-p
    np.testing.assert_almost_equal(np.array(sum([z**2, z*p, z*p, r])), np.array([1]))

    mu = np.array([
        z**6,                                       #0
        2*3*z**5 * p,                               #1
          3*z**4 *        r,                        #2

        2*3*z**4 * p**2,                            #4
          3*z**4 * p**2,                            #3  convergent
          3*z**4 * p**2,                            #5  divergent

        2*3*z**3 * p**3,                            #7  feed-forward loop
        2*3*z**3 * p  *r,                           #8
        2*3*z**3 * p    * r,                        #6
        2*  z**3 * p**3,                            #9  feedback loop

          3*z**2 *        r**2,                     #11
        2*3*z**2 * p**2 * r,                        #12
          3*z**2 * p**2 * r,                        #13
          3*z**2 * p**2 * r,                        #10

        2*3*z*p  *        r**2,                     #14
                          r**3                      #15
    ])
    return mu


def triad_classification_tree():
    def permute(X, order):
        order = np.array(order)
        return np.array([row[order] for row in X[order]])
    motifs = triad_patterns()

    tags = -1*np.ones(shape=(4,4,4), dtype=np.int)
    permutations = list(itertools.permutations(range(3), 3))
    for i in range(len(motifs)):
        motif = motifs[i]
        for perm in permutations:
            isomporth = permute(motif, perm)
            edges = isomporth + 2*isomporth.T
            tags[edges[0,1], edges[0,2], edges[1,2]] = i

    return tags


def triad_classification_tree_with_participation():
    def permute(X, order):
        order = np.array(order)
        return np.array([row[order] for row in X[order]])
    motifs = triad_patterns()

    tags = -1*np.ones(shape=(4,4,4,4), dtype=np.int)
    permutations = list(itertools.permutations(range(3), 3))
    permutations = [(p, np.argsort(p)) for p in permutations]
    for i in range(len(motifs)):
        motif = motifs[i]
        for perm, perm_inv in permutations:
            isomporth = permute(motif, perm)
            edges = isomporth + 2*isomporth.T
            tags[edges[0,1], edges[0,2], edges[1,2],:] = np.hstack([i, perm_inv])

    return tags


# np.testing.assert_equal( triads_classification_tree(), triads_classification_tree_old() )
# %timeit triads_classification_tree_old()
# %timeit triads_classification_tree()



# Compatibility with several conventions
triad_order_bct = 3 + np.array([1, 0, 2, 5, 3, 4, 6, 10, 7, 8, 9, 11, 12])              # j.neuroimage.2009.10.003
triad_order_egger2014 = 3 + np.array([12, 6, 11, 8, 9, 10, 3, 7, 0, 4, 5, 1, 2])        # fnana.2014.00129
triad_order_nn4576 = 3 + np.arange(13)                                                  # nn.4576


def index_all(elements, array):
    return np.array([np.where(array==x)[0][0] for x in elements])


conv_triad_order_nn4576_to_bct = index_all(triad_order_bct, triad_order_nn4576)
assert np.array_equal(triad_order_nn4576[conv_triad_order_nn4576_to_bct], triad_order_bct)
conv_triad_order_nn4576_to_egger2014 = index_all(triad_order_egger2014, triad_order_nn4576)
assert np.array_equal(triad_order_nn4576[conv_triad_order_nn4576_to_egger2014], triad_order_egger2014)


def identify_triad_node_roles():
    triads = triad_patterns()

    node_roles = []
    for i in range(len(triads)):
        triad = triads[i]

        triad_node_roles = [0, 1, 2]
        if np.array_equal(triad, triad[np.ix_([1, 0, 2], [1, 0, 2])]):
            triad_node_roles[1] = triad_node_roles[0]
        if np.array_equal(triad, triad[np.ix_([2, 1, 0], [2, 1, 0])]):
            triad_node_roles[2] = triad_node_roles[0]
        if np.array_equal(triad, triad[np.ix_([0, 2, 1], [0, 2, 1])]):
            triad_node_roles[2] = triad_node_roles[1]

        node_roles.append(triad_node_roles)

    return node_roles
