# netsci
Analyzing Complex Networks with Python


|  Author   |                                        Version                                        |                                                                     Demo                                                                      |
| :-------: | :-----------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------: |
| Gialdetti | [![PyPI](https://img.shields.io/pypi/v/netsci.svg)](https://pypi.org/project/netsci/) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gialdetti/netsci/master?filepath=examples%2Fnetwork_motifs.ipynb) |


`netsci` is a python package for efficient statistical analysis of spatially-embedded networks. In addition, it offers several algorithms and implementations (CPU and GPU-based) of motif counting algorithms.

For other models and metrics, we highly recommend using existing and richer tools. Noteworthy packages are the magnificent [NetworkX](https://networkx.github.io), [graph-tool](https://graph-tool.skewed.de) or [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/).


## A simple example
Analyzing a star network (of four nodes)

```python
import numpy as np
import netsci.visualization as nsv

A = np.array([[0,1,1,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]])
nsv.plot_directed_network(A, pos=[[0,0],[-1,1],[1,1],[0,-np.sqrt(2)]])
```
![Alt text](./examples/images/star4_network.png)


```python
import netsci.metrics.motifs as nsm
f = nsm.motifs(A, algorithm='brute-force')
print(f)
# [1 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0]
```

```python
nsv.bar_motifs(f)
```
![Alt text](examples/images/star4_motifs.png)

## GPU speedup
Using GPU for the motif counting is easy
```python
from netsci.models.random import erdos_renyi

# Create an Erdős–Rényi network, and count motifs using a GPU
A_er = erdos_renyi(n=500, p=0.2, random_state=71070)  
f_er = nsm.motifs(A_er, algorithm="gpu")

# Visualize
print(f_er)
# [5447433 8132356 1031546 2023563 1011703 1011109  503098  512458
#   513352  167427   64844  127751   64442   63548   32483    1387]
nsv.bar_motifs(f_er)
```
![](examples/images/er_motifs.png)

The running-time speedup ratio resulting from the GPU-based implementation, as measured over several networks sizes (n) and sparsities (p), is depicted below

![](examples/images/gpu-speedup-times.TeslaT4.(250821.205715).png)

A full a live notebook for performing this benmarching is provided [below](#examples).

## Installation
### Install latest release version via [pip](https://pip.pypa.io/en/stable/quickstart/)
```bash
$ pip install netsci
```

### Install latest development version
via pip
```bash
$ pip install git+https://github.com/gialdetti/netsci.git
``` 
or in development mode
```bash
$ git clone https://github.com/gialdetti/netsci.git
$ cd netsci
$ pip install -e .[dev]
```

## Testing
After installation, you can launch the test suite:
```bash
$ pytest
```


## Help and Support

### Examples

| Theme                                                                                                                                                      |                                                                          MyBinder                                                                           |                                                                                              Colab                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Basic network motifs demo](https://nbviewer.org/github/gialdetti/netsci/blob/master/examples/network_motifs.ipynb)                                        |        [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gialdetti/netsci/master?filepath=examples%2Fnetwork_motifs.ipynb)        |                                                                                                                                                                                                 |
| [Connectomics dataset, and 3-neuron motif embedding](https://nbviewer.org/github/gialdetti/netsci/blob/master/examples/connectomics_motif_embedding.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gialdetti/netsci/master?filepath=examples%2Fconnectomics_motif_embedding.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gialdetti/netsci/blob/master/examples/connectomics_motif_embedding.ipynb) |
| Tech: [GPU speedup of motif analysis](https://nbviewer.org/github/gialdetti/netsci/blob/master/examples/tech/gpu_speedup.ipynb)                            |      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gialdetti/netsci/master?filepath=examples%2Ftech%2Fgpu_speedup.ipynb)      |       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gialdetti/netsci/blob/master/examples/tech/gpu_speedup.ipynb)       |


### Communication
Please send any questions you might have about the code and/or the algorithm to <eyal.gal@mail.huji.ac.il>.


### Citation
If you use `netsci` in a scientific publication, please consider citing the following paper:

> Gal, E., Perin, R., Markram, H., London, M., and Segev, I. (2019). [Neuron Geometry Underlies a Universal Local Architecture in Neuronal Networks.](https://doi.org/10.1101/656058) BioRxiv 656058.

Bibtex entry:

    @article {Gal2019
        author = {Gal, Eyal and Perin, Rodrigo and Markram, Henry and London, Michael and Segev, Idan},
        title = {Neuron Geometry Underlies a Universal Local Architecture in Neuronal Networks},
        year = {2019},
        doi = {10.1101/656058},
        journal = {bioRxiv}
    }
