import logging
import os
import time

import numpy as np
import pandas as pd

RESOURCES_ROOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../resources")
)
logging.info(f'RESOURCES_ROOT_PATH = "{RESOURCES_ROOT_PATH}"')


def load_connectome(celltype="L5_TTPC", adjacency=False):
    """Load and return the connectome dataset

    The connectome dataset describes the network of synaptic connections among neurons from a rat cortex. There are several 
    cell-type-specific subcircuits that their detailed connectivity is available.

    Parameters
    ----------
    celltype : str, default="L5_TTPC"
        The cell-type defining the required subcircuit.
    adjacency : boolean, default=False.
        If True, returns also the binary (`A`) and weighted (`W`) adjacency matrices. See below for more information about the
        `A` and `W` object.

    Returns
    -------
    connectome : dictionary
        `nodes`: DataFrame
            Listing all neurons and their properties. Each neuron is described by its id and its 3d position.
        `edges`: DataFrame
            Listing all connections and their properties. Each connection is described by its source
            neuron, target neuron and the number of contacts (synapses) realizing this connection.
        `A`: 2d-array, optional
            The binary adjacency matrix.
        `W`: 2d-array, optional
            The weighted adjacency metrics, whose each entry depicts the number of synapses in the respective
            connection.

    References
    --------
    * Markram, H., Muller, E., Ramaswamy, S., Reimann, M.W., Abdellah, M., Sanchez, C.A., et al. (2015). Reconstruction
      and Simulation of Neocortical Microcircuitry. Cell 163, 456-492.

    """

    neurons = pd.read_csv(get_resource_path(f"datasets/connectome.{celltype}.neurons.csv.gz"))
    synapses = pd.read_csv(get_resource_path(f"datasets/connectome.{celltype}.synapses.csv.gz"))
    connectome = dict(title=f"Connectome ({celltype.replace('_', '-')}", nodes=neurons, edges=synapses)

    if adjacency:
        W = edges_to_adjacency(synapses, nodes=neurons["gid"])
        connectome["A"] = (W != 0).astype(int)
        connectome["W"] = W

    return connectome


"""
Utility functions
"""


def get_resource_path(sub_path):
    return os.path.join(
        RESOURCES_ROOT_PATH, sub_path.replace("{ts}", time.strftime("(%y%m%d.%H%M%S)"))
    )


def edges_to_adjacency(edges, nodes=None, source="from", target="to", value="contacts"):
    if nodes is None:
        nodes = sorted(
            np.unique(np.hstack([edges[source].unique(), edges[target].unique()]))
        )

    A = (
        edges.pivot_table(index=source, columns=target, values=value, fill_value=0)
        .reindex(nodes, axis=0, fill_value=0)
        .reindex(nodes, axis=1, fill_value=0)
        .values
    )

    return A
