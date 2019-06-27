# netsci
Analyzing Complex Networks with Python


netsci is a python package for efficient statistical analysis of spatially-embedded networks. In addition, it offers efficient implementations of motif counting algorithms.
For other models and metrics, we highly recommend using existing and richer tools. Noteworthy packages are the magnificent [NetworkX](https://networkx.github.io), [graph-tool](https://graph-tool.skewed.de) or [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/).

## Simple example
Analyzing a star network (of four nodes)

```python
>>> import numpy as np
>>> import netsci.visualization as nsv
>>> A = np.array([[0,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]])
>>> nsv.plot_directed_network(A, pos=[[0,0],[-1,1],[1,1],[0,-np.sqrt(2)]])
```
![Alt text](./examples/images/star4_network.png)


```python
>>> import netsci.metrics.motifs as nsm
>>> f = nsm.motifs(A, algorithm='brute-force')
>>> print(f)
[1 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0]
```

```python
>>> nsv.bar_motifs(f)
```
![Alt text](examples/images/star4_motifs.png)


## Testing
After installation, you can launch the test suite:
```python
$ pytest
```
