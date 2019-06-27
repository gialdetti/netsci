from matplotlib import pyplot as plt, gridspec

from netsci.metrics.motifs import *


default_triad_order = triad_order_nn4576


def bar_motifs(bar, line=None, order=None, title=None):
    if order is None:
        order = range(len(bar)) if bar[0]>-1 else default_triad_order

    x = 1 + np.arange(len(order))

    plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 3])
    ax0 = plt.subplot(gs[0])
    ax0.bar(x, bar[order])
    if line is not None:
        ax0.plot(x, line[order], '-oy')
    print(x)
    plt.xticks(x)
    if title is not None:
        plt.title(title)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    plot_all_triads(order, ax=ax1, label=False)
    plt.axis('off')

    return [ax0, ax1]


def plot_a_triad(tid, o=(0, 0), r=1, phi=np.pi / 2, label=True,
                 arrow_color=None, head_width=0.10,
                 nodes=True, node_color=None, node_size=12, node_alpha=.5,
                 bg_color=None, ax=None):
    motif = triad_patterns()[tid] if np.isscalar(tid) else tid
    xy = o+r*np.array([[np.cos(-2*np.pi*a+phi), np.sin(-2*np.pi*a+phi)] for a in np.arange(3)/3.0])

    scale = 0.75

    ax = ax or plt.gca()
    arrow_color = arrow_color or ax.get_xticklabels()[0].get_color()

    # ax.plot(xy.T[0],xy.T[1],'.')

    if nodes:
        ax.scatter(xy.T[0], xy.T[1], color=node_color, s=node_size, alpha=node_alpha)

    h = [ax.arrow(xy[i,0], xy[i,1],
                  scale*(xy[j,0]-xy[i,0]), scale*(xy[j,1]-xy[i,1]),
                  head_width=head_width, color=arrow_color, fc=arrow_color, ec=arrow_color)
         for (i, j) in zip(*motif.nonzero())]
    if label is not False:
        if label is True:
            label = tid
        ax.text(o[0], o[1], "%s" % label, horizontalalignment="center", verticalalignment="center", fontsize=8)

    if bg_color:
        bg = plt.Polygon(xy, color=bg_color)
        ax.add_patch(bg)

    return h


# TODO: check https://stackoverflow.com/questions/8733558/how-can-i-make-the-xtick-labels-of-a-plot-be-simple-drawings-using-matplotlib
def plot_all_triads(order=default_triad_order, o=(1, 0), delta=(1, 0), r=.4, **kwargs):
    [plot_a_triad(order[i], o=np.array(o) + np.array([i * delta[0], np.mod(i, 2) * delta[1]]), r=r, **kwargs) for i in range(len(order))]
    plt.axis('equal')



