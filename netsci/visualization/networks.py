from matplotlib import pyplot as plt, gridspec

from scipy.spatial.distance import pdist, squareform
import numpy as np


def plot_network(A=None, pos=None):
    """

    Parameters
    ----------
    A
    pos

    Returns
    -------

    """

    if A is None:
        p_d = lambda D: D < 0.112
        n = 400
        pos = np.random.rand(n, 2)
        A = np.random.binomial(1, p_d(squareform(pdist(pos))))

    node_color, edge_color = 'k', [.6]*3
    # node_color, edge_color = np.array([100, 125, 216])/255, np.array([154, 218, 238])/255
    pos = np.array(pos)

    pre_xy, post_xy = [np.array([pos[x] for x in indices]) for indices in A.nonzero()]
    xs = np.array([pre_xy[:,0],post_xy[:,0]])
    ys = np.array([pre_xy[:,1],post_xy[:,1]])

    plt.figure(figsize=(5,5))
    plt.scatter(pos[:, 0], pos[:, 1], color=node_color, zorder=2)
    plt.axis("square")
    # plt.autoscale(enable=True, tight=True)
    plt.plot(xs,ys,'-', lw=.5, color=edge_color, zorder=1)
    plt.title("|V| = %d, |E| = %d" % (A.shape[0], A.sum()))


default_node_color = np.array([116, 95, 169])/255
default_edge_color = np.array([84, 88, 87])/255


def plot_directed_network(A=None, pos=None, title=None, labels=None,
                          fig_kws=dict(figsize=(5, 5)), node_color=default_node_color, edge_color=default_edge_color):
    """

    Parameters
    ----------
    A
    pos
    colors
    title
    labels
    fig_kws

    Returns
    -------

    """
    if A is None:
        n = 50
        pos = np.random.rand(n, 2) * 2 - 1
        foo = lambda u, v: np.all(np.sign(u) == np.sign(v)) and \
                           np.all(np.sum(u**2) > np.sum(v**2))
                           # np.all(np.abs(u) > np.abs(v))

        p = np.array([[foo(u, v) for v in pos] for u in pos]) * \
            (squareform(pdist(pos)) < .5)
        A = np.random.binomial(1, p)

    pos = np.array(pos)

    colors = [node_color] * len(pos)
    plt.figure(**fig_kws)
    plt.scatter(pos[:, 0], pos[:, 1], c=colors, s=200, alpha=1, zorder=10)
    plt.axis('square')
    [plt.annotate('', xy=pos[i], xytext=pos[j],
                  arrowprops=dict(facecolor=edge_color, edgecolor=edge_color, arrowstyle='<|-, head_length=1, head_width=.4'))
     # arrowprops=dict(facecolor=arrow_color, edgecolor=arrow_color, shrink=0.07))
     for i, j in np.array(A.nonzero()).T]
    labels = labels or range(len(A))
    [plt.text(x, y, label, color='w', ha='center', va='center', zorder=20, fontsize=8)
     for label, (x, y) in zip(labels, pos[:, :2])]
    if title is not None:
        plt.title(title)
