import numpy as np
import matplotlib.pyplot as plt

import netsci.metrics.motifs as nsm
import netsci.visualization as nsv


node_roles = nsm.identify_triad_node_roles()

rgb = list('rgb')
rgb = np.array([[225, 96, 67], [228, 146, 82], [0, 148, 203]])/255.0

plt.figure(figsize=(13, 1))
[nsv.plot_a_triad(i, o=(1+i,0), r=.4, node_size=75, ax=plt.gca(), label=i,
                  node_color=np.take(rgb, node_roles[i], axis=0))
 for i in range(len(node_roles))]

plt.axis('equal')
plt.axis('off')
