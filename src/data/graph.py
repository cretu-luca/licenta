import ast

import numpy as np
import snappy
import torch
from torch_geometric.data import Data


NODE_TYPE_CROSSING_NEG = 0
NODE_TYPE_CROSSING_POS = 1
NODE_TYPE_MIDPOINT = 2
NODE_TYPE_UNKNOT = 3
NUM_NODE_TYPES = 4


def fix_pd_notation(pd_notation: str) -> str:
    return pd_notation.replace(";", ",")


def pd_to_crossing_graph(pd_notation: str):
    pd_list = ast.literal_eval(fix_pd_notation(pd_notation))
    link = snappy.Link(pd_list)
    crossings = link.crossings

    if len(crossings) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.array([NODE_TYPE_UNKNOT], dtype=np.int64),
        )

    num_crossings = len(crossings)

    node_types: list[int] = [
        NODE_TYPE_CROSSING_POS if c.sign == 1 else NODE_TYPE_CROSSING_NEG
        for c in crossings
    ]

    sources: list[int] = []
    targets: list[int] = []

    next_midpoint_idx = num_crossings

    for crossing in crossings:
        for pos in range(4):
            neighbor, neighbor_pos = crossing.adjacent[pos]

            if (crossing.label, pos) < (neighbor.label, neighbor_pos):
                midpoint = next_midpoint_idx
                next_midpoint_idx += 1

                node_types.append(NODE_TYPE_MIDPOINT)

                sources.extend([crossing.label, midpoint])
                targets.extend([midpoint, neighbor.label])

    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(node_types, dtype=np.int64),
    )


def pd_to_crossing_graph_data(pd_notation: str, y: torch.Tensor | None = None) -> Data:
    sources, targets, node_types = pd_to_crossing_graph(pd_notation)

    edge_index = torch.from_numpy(
        np.stack(
            [
                np.concatenate([sources, targets]),
                np.concatenate([targets, sources]),
            ]
        )
    ).to(torch.long)

    x = torch.from_numpy(node_types).to(torch.long).unsqueeze(-1)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=node_types.shape[0],
    )