"""Graph -> CW lifting that consumes user-supplied 2-cells.

TopoBench's stock graph2cell liftings (`cycle`, `discrete_configuration_complex`)
auto-detect faces from graph structure. Knot diagrams supply faces explicitly
(via SnapPy's planar embedding), so this transform reads `data.faces`, builds a
`toponetx.CellComplex`, and populates incidence/laplacian/adjacency tensors via
TopoBench's `get_complex_connectivity`.

Also fills `x_0` (one-hot of node types from `data.x`), `x_1`, `x_2`. Drops
`data.faces` after consumption so PyG batching never has to collate the
variable-length list-of-lists.
"""
from __future__ import annotations

import torch
from torch_geometric.transforms import BaseTransform
from toponetx import CellComplex
from topobench.data.utils.utils import (
    generate_zero_sparse_connectivity,
    get_complex_connectivity,
)

from src.data.knot import NUM_NODE_TYPES


class Graph2CellFaceLifting(BaseTransform):
    """Builds CW-complex tensors from a graph + user-supplied 2-cell list."""

    def __init__(
        self,
        complex_dim: int = 2,
        signed: bool = False,
        num_node_types: int = NUM_NODE_TYPES,
    ) -> None:
        self.complex_dim = int(complex_dim)
        self.signed = bool(signed)
        self.num_node_types = int(num_node_types)

    def forward(self, data):
        node_types = data.x.view(-1).to(torch.long)
        x_0 = torch.nn.functional.one_hot(
            node_types, num_classes=self.num_node_types
        ).to(torch.float)
        data.x_0 = x_0

        faces = list(getattr(data, "faces", []) or [])
        edge_index = data.edge_index
        num_edges = int(edge_index.size(1) // 2)
        num_faces = int(len(faces))

        if num_edges == 0 and num_faces == 0:
            data.x_1 = torch.zeros((0, 1), dtype=torch.float)
            data.x_2 = torch.zeros((0, 1), dtype=torch.float)
            n0 = int(data.num_nodes)
            for r in range(self.complex_dim + 1):
                m = n0 if r == 0 else 0
                data[f"incidence_{r}"] = generate_zero_sparse_connectivity(m=m, n=m)
                for k in (
                    "down_laplacian",
                    "up_laplacian",
                    "adjacency",
                    "coadjacency",
                    "hodge_laplacian",
                ):
                    data[f"{k}_{r}"] = generate_zero_sparse_connectivity(m=m, n=m)
            data.shape = [n0, 0, 0]
        else:
            data.x_1 = torch.ones((num_edges, 1), dtype=torch.float)
            data.x_2 = torch.ones((num_faces, 1), dtype=torch.float)

            cc = CellComplex()
            for node_idx in range(int(data.num_nodes)):
                cc.add_node(node_idx)
            seen_edges: set[tuple[int, int]] = set()
            for u, v in edge_index.t().tolist():
                key = (u, v) if u <= v else (v, u)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                cc.add_edge(int(u), int(v))
            for face in faces:
                cc.add_cell(list(face), rank=2)

            conn = get_complex_connectivity(cc, self.complex_dim, signed=self.signed)
            for k, v in conn.items():
                data[k] = v

        if hasattr(data, "faces"):
            del data.faces
        return data

    def __call__(self, data):
        return self.forward(data)
