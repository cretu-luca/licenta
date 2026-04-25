"""SnapPy → PyG: parse PD codes into the subdivided knot-diagram CW complex.

Domain-specific code that no library can replace. Emits a plain `Data(x,
edge_index, faces, y, num_crossings, ...)`. CW-connectivity tensors
(`x_0`, `incidence_*`, `down_laplacian_*`, ...) are produced downstream by
`src.transforms.graph2cell_face_lifting.Graph2CellFaceLifting`.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import snappy
import torch
from torch_geometric.data import Data

NODE_TYPE_CROSSING_NEG = 0
NODE_TYPE_CROSSING_POS = 1
NODE_TYPE_MIDPOINT = 2
NODE_TYPE_UNKNOT = 3
NUM_NODE_TYPES = 4


def _fix_pd_notation(pd_notation: str) -> str:
    return pd_notation.replace(";", ",")


def _arc_key(label_a: int, port_a: int, label_b: int, port_b: int) -> frozenset:
    return frozenset([(label_a, port_a), (label_b, port_b)])


@dataclass(frozen=True)
class KnotDiagramTopology:
    """Combinatorial data of a subdivided knot-diagram CW complex.

    Node layout: [crossings (0..n-1), midpoints (n..n+2n-1)].
    """

    node_types: np.ndarray
    edges: np.ndarray
    faces: list[list[int]]
    num_crossings: int

    @property
    def num_nodes(self) -> int:
        return int(self.node_types.shape[0])

    @classmethod
    def from_pd(cls, pd_notation: str) -> "KnotDiagramTopology":
        """Parse a PD code into the subdivided 1-skeleton plus face boundaries.

        Uses SnapPy's planar embedding to extract faces; each arc gets a unique
        midpoint node, so the same midpoint index appears in both the 1-skeleton
        edges and the face it bounds.
        """
        pd_list = ast.literal_eval(_fix_pd_notation(pd_notation))
        link = snappy.Link(pd_list)
        crossings = link.crossings

        if len(crossings) == 0:
            return cls(
                node_types=np.array([NODE_TYPE_UNKNOT], dtype=np.int64),
                edges=np.empty((0, 2), dtype=np.int64),
                faces=[],
                num_crossings=0,
            )

        num_crossings = len(crossings)
        node_types: list[int] = [
            NODE_TYPE_CROSSING_POS if c.sign == 1 else NODE_TYPE_CROSSING_NEG
            for c in crossings
        ]

        arc_to_midpoint: dict[frozenset, int] = {}
        arc_endpoints: dict[frozenset, tuple[int, int]] = {}
        next_midpoint_idx = num_crossings

        for crossing in crossings:
            for port in range(4):
                neighbor, neighbor_port = crossing.adjacent[port]
                key = _arc_key(crossing.label, port, neighbor.label, neighbor_port)
                if key not in arc_to_midpoint:
                    arc_to_midpoint[key] = next_midpoint_idx
                    arc_endpoints[key] = (crossing.label, neighbor.label)
                    next_midpoint_idx += 1
                    node_types.append(NODE_TYPE_MIDPOINT)

        edges: list[tuple[int, int]] = []
        for key, midpoint in arc_to_midpoint.items():
            crossing_a, crossing_b = arc_endpoints[key]
            edges.append((crossing_a, midpoint))
            edges.append((midpoint, crossing_b))

        faces: list[list[int]] = []
        for strand_cycle in link.faces():
            face_boundary: list[int] = []
            for strand in strand_cycle:
                c = strand.crossing
                e = strand.strand_index
                next_port = (e + 1) % 4
                neighbor, neighbor_port = c.adjacent[next_port]
                arc_key_here = _arc_key(
                    c.label, next_port, neighbor.label, neighbor_port
                )
                midpoint = arc_to_midpoint[arc_key_here]
                face_boundary.append(c.label)
                face_boundary.append(midpoint)
            faces.append(face_boundary)

        return cls(
            node_types=np.asarray(node_types, dtype=np.int64),
            edges=np.asarray(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64),
            faces=faces,
            num_crossings=num_crossings,
        )

    def topology_to_pyg_data(
        self,
        y: torch.Tensor | None = None,
        knot_name: str | None = None,
        pd_notation: str | None = None,
    ) -> Data:
        """Emit a plain PyG Data; CW connectivity is built later by the lifting."""
        if self.edges.shape[0] > 0:
            src = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
            dst = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
            edge_index = torch.from_numpy(np.stack([src, dst])).to(torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        x = torch.from_numpy(self.node_types).to(torch.long).unsqueeze(-1)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=self.num_nodes,
        )
        data.faces = [list(map(int, face)) for face in self.faces]
        data.num_crossings = torch.tensor([int(self.num_crossings)], dtype=torch.long)
        if knot_name is not None:
            data.knot_name = str(knot_name)
        if pd_notation is not None:
            data.pd_notation = str(pd_notation)
        return data
