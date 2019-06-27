import numpy as np

from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt

import netsci.metrics.motifs as nsm
import netsci.visualization.motifs as nsv


def radial_scale(x,y, by):
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    rho, phi = cart2pol(x, y)
    x, y = pol2cart(by*rho, phi)
    return x, y


tree = nsm.triad_classification_tree()

G = nx.Graph()
DG = nx.DiGraph()


def add_edge(u, v, type):
    if type==1 or type==3:
        DG.add_edge(u,v, type=type)
    if type==2 or type==3:
        DG.add_edge(v,u, type=type)


l0 = "r"
for uv in range(tree.shape[0]):
    l1 = "%d" % uv
    print (l0, l1)
    G.add_edge(l0, l1, type=uv)
    add_edge(l0, l1, type=uv)
    for uw in range(tree.shape[1]):
        l2 = "%02d" % (uv * 10 + uw)
        print("\t", (l1, l2))
        G.add_edge(l1, l2, type=uw)
        add_edge(l1, l2, type=uw)
        for vw in range(tree.shape[2]):
            l3 = "%03d" % (uv * 100 + uw * 10 + vw)
            print("\t\t", (l2, l3))
            G.add_edge(l2, l3, type=vw)
            add_edge(l2, l3, type=vw)

            mid = tree[uv, uw, vw]
            print("\t\t\t #%d" % mid)
            G.add_node(l3, mid=mid)



pos = graphviz_layout(G, prog='twopi', args='')

plt.figure(figsize=(8, 8))
# nx.draw(G, pos=pos, alpha=.5,
#         with_labels=False, edge_color=[[.7,.7,.7]]*len(G.edges()))
# nx.draw(DG, pos=pos)
[ox, oy] = np.mean(list(pos.values()), axis=0)
pos = {k: (x-ox,y-oy) for k,(x,y) in pos.items()}

nx.draw_networkx_edges(G, pos, alpha=.2)
nx.draw_networkx_edges(DG, pos=pos,
                       edgelist=DG.edges(data=True),
                       edge_color=plt.cm.Set1([3-d['type'] for (u,v,d) in DG.edges(data=True)]))
plt.axis('equal')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


labels = dict([(u,d['mid']) for u,d in G.nodes(data=True) if 'mid' in d])
# nx.draw_networkx_labels(G, pos, labels=labels)
edge_labels = dict([((u,v),d['type']) for u,v,d in G.edges(data=True)])
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

scaled_pos = {k: radial_scale(x,y, 1.07) for k,(x,y) in pos.items()}
[nsv.plot_a_triad(mid, scaled_pos[u], r=9, label=False, head_width=3,
                  nodes=not False, node_color=['r', 'g', 'b'],
                  bg_color=plt.cm.tab20c_r(mid))
 for (u, mid) in labels.items()]


if False:
    import itertools

    A = np.random.randint(2, size=(3,3)) * (np.eye(3)==0)
    # A = array([[0,0,0], [1,0,1], [1,0,0]])
    f = nsm.motifs_naive(A)
    assert sum(f)==1
    mid = np.where(f==1)[0][0]

    print("A =\n", A, "#%d" % mid)
    M = nsm.triads_patterns()[mid]
    print("M =\n", M, np.array_equal(A, M))


    def permute(X, order):
        order = np.array(order)
        return np.array([row[order] for row in X[order]])
    m2a = np.array(next(
        m2a for m2a in itertools.permutations(range(3), 3) if np.array_equal(A, permute(M, m2a))))
    print(m2a)
    a2m = np.array(next(
        a2m for a2m in itertools.permutations(range(3), 3) if np.array_equal(M, permute(A, a2m))))
    print(a2m)

    RGB = np.array(list('RGB'))
    wuv = np.array(list('wuv'))
    f, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(6,2))
    nsv.plot_a_triad(A,   r=0.5, ax=ax[0], node_size=96, label=False), ax[0].set_title(",".join(["%s=?"%u for u in wuv]))
    nsv.plot_a_triad(mid, r=0.5, ax=ax[1], node_size=96, node_color=RGB, bg_color=[.7]*3), ax[1].set_title("Motif (%s)"%",".join(RGB))
    ordered_RGB = RGB[m2a]
    nsv.plot_a_triad(A,   r=0.5, ax=ax[2], node_size=96, node_color=ordered_RGB, label=False), ax[2].set_title(",".join(["%s=%s"%(u,r) for (u,r) in zip(wuv,ordered_RGB)]))

    print("RGB[m2a] =", RGB, m2a, "=", RGB[m2a], "\t# colors", wuv)
    print("wuv[a2m] =", wuv, a2m, "=", wuv[a2m], "\t# RGB-ordered gids")
