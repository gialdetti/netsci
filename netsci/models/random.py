import numpy as np


def erdos_renyi(n, p):
    """Creates a directed Erdős–Rényi random network

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        The marginal probability for a directed edge.

    Returns
    -------
    A : 2d-array
        Binary adjacency matrix.

    See Also
    --------
    erdos_renyi_reciprocal

    """
    # E[Xij] = p
    #
    # Maximal randomness - pairs (i,j) are i.i.d. and define by
    #       Xij ~ Bernoulli(p)
    #
    # Note: related to Max-Ent / ERGM constrained by E_X[f(X)]=p, where f(X):=E_Xij[Xij]

    A = np.random.binomial(1, p, size=(n, n))
    np.fill_diagonal(A, 0)
    return A


def erdos_renyi_reciprocal(n, p, r):
    """Creates a directed Erdős–Rényi-like random network extended with a reciprocity constraint

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        The marginal probability for a directed edge.
    r : float
        The marginal probability for a reciprocal edge.

    Returns
    -------
    A : 2d-array
        Binary adjacency matrix.

    See Also
    --------
    erdos_renyi_reciprocal

    """

    """the Math
    
    E[Xij]    = p
    E[XijXji] = r

    Maximal-randomness (?) solution: (unordered) pairs {i,j} are i.i.d. and defined by

          Y{ij} = XijXji ~ { 0:  -  (0,0)  :  p_none             := 1 - 2p_uni - p_recip
                             1:  -> (0,1)  :  p_uni   := p - r
                             2: <-  (1,0)  :  p_uni   := p - r
                             3: <-> (1,1)  :  p_recip := r

    Check:
      E[XijXji] = Pr(XijXji=1) = Pr(Xij==1 and Xji == 1) = Pr(Yij=3) = r
      E[Xij]    = Pr(Xij=1)                              = Pr(Yij=2) + Pr(Yij=3) = p_uni + r = p
    """

    p_recip = r
    p_uni = p - r
    p_none = 1 - 2*p_uni - p_recip

    Y = np.triu(np.random.choice(4, size=(n, n), p=[p_none, p_uni, p_uni, p_recip]),1)
    A_bi = np.int8(Y==3)
    A = np.int8(Y==1) + np.int8(Y.T==2) + (A_bi + A_bi.T)

    return A
