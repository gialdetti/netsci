import logging
import warnings

import numpy as np
import numpy.testing as npt

from netsci.datasets import load_connectome


def test_load_connectome():
    connectome = load_connectome(adjacency=True)

    assert 'nodes' in connectome
    assert 'edges' in connectome
    assert 'A' in connectome
    assert 'W' in connectome


def test_load_connectome_with_adjacency():
    connectome = load_connectome(adjacency=True)

    assert 'nodes' in connectome
    assert 'edges' in connectome
    assert 'A' in connectome
    assert 'W' in connectome


def test_consistency_with_bb():
    try: 
        from bbpy import BBModelInstance, Cluster
    except ImportError:
        warnings.warn('The bbpy package cannot be imported. Skipping test!', ImportWarning)
        return

    connectome = load_connectome(adjacency=True)
    neurons, synapses, A, W = connectome['nodes'], connectome['edges'], connectome['A'], connectome['W']

    hcid, structural, cell_type = 2, False, 'L5_TTPC2'
    subpopulation = Cluster.get(cell_type)
    logging.info('Cluster: %s' % subpopulation.title)

    bb = BBModelInstance(hcid)
    W0 = bb.cmat(subpopulation, subpopulation).todense()
    A0 = (W0!=0).astype(np.int8)
    pos = bb.coords(subpopulation)
    gids = bb.gids(subpopulation)

    logging.info(f'{subpopulation.name} connectivity: {A.shape[0]}x{A.shape[1]}')

    npt.assert_array_equal(A0, A)
    npt.assert_array_equal(W0, W)
    