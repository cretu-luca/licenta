import ast
from dataclasses import dataclass

import numpy as np
import snappy
import torch
from torch_geometric.data import Data

import toponetx as tnx

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

    Attributes
    ----------
    node_types : (num_nodes,) int64
        One of NODE_TYPE_*.
    edges : (num_edges, 2) int64
        Undirected edges of the subdivided 1-skeleton, one row per undirected edge.
        Each arc contributes exactly two rows: (crossing_a, midpoint) and
        (midpoint, crossing_b).
    faces : list of list of int
        Each face is a cyclic sequence of node indices alternating crossing,
        midpoint, crossing, midpoint, ..., suitable for
        `CellComplex.add_cell(face, rank=2)`.
    num_crossings : int
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

        # Degenerate case: the 0-crossing unknot. No arcs, no faces in the usual sense.
        # Represent it as a single isolated node. Batch collation works with empty
        # edge_index if num_nodes is set.
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

        # Allocate a midpoint per arc. Traverse in a deterministic order so that
        # indices are stable across runs.
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

        # Build the edge list: each arc contributes two edges (crossing -- midpoint,
        # midpoint -- other_crossing). Self-loop arcs (same crossing both ends) still
        # produce two well-defined edges through the unique midpoint.
        edges: list[tuple[int, int]] = []
        for key, midpoint in arc_to_midpoint.items():
            crossing_a, crossing_b = arc_endpoints[key]
            edges.append((crossing_a, midpoint))
            edges.append((midpoint, crossing_b))

        # Extract faces from SnapPy's planar embedding. Each face is a cyclic list of
        # CrossingStrands; each strand (c, p) leaves crossing c through port p along
        # some arc, landing at strand.opposite(). The face boundary in the subdivided
        # graph alternates: crossing -> midpoint -> next crossing -> midpoint -> ...
        faces: list[list[int]] = []
        for strand_cycle in link.faces():
            face_boundary: list[int] = []
            for strand in strand_cycle:
                opposite = strand.opposite()
                arc_key_here = _arc_key(
                    strand.crossing.label,
                    strand.strand_index,
                    opposite.crossing.label,
                    opposite.strand_index,
                )
                midpoint = arc_to_midpoint[arc_key_here]
                face_boundary.append(strand.crossing.label)
                face_boundary.append(midpoint)
            faces.append(face_boundary)

        return cls(
            node_types=np.asarray(node_types, dtype=np.int64),
            edges=np.asarray(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64),
            faces=faces,
            num_crossings=num_crossings,
        )


    def topology_to_pyg_data(self, y: torch.Tensor | None = None) -> Data:
        if self.edges.shape[0] > 0:
            src = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
            dst = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
            edge_index = torch.from_numpy(np.stack([src, dst])).to(torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        x = torch.from_numpy(self.node_types).to(torch.long).unsqueeze(-1)

        return Data(x=x, edge_index=edge_index, y=y, num_nodes=self.num_nodes)


    def topology_to_cell_complex(self) -> tnx.CellComplex:
        cell_complex = tnx.CellComplex()

        for node_idx in range(self.num_nodes):
            cell_complex.add_node(node_idx)

        for u, v in self.edges:
            cell_complex.add_edge(int(u), int(v))

        for face in self.faces:
            cell_complex.add_cell(face, rank=2)

        return cell_complex