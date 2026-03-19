import ast
import math
import os
import numpy as np
import torch
import pandas as pd
import snappy
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


def fix_pd_notation(pd_notation):
    return pd_notation.replace(';', ',')


def pd_to_crossing_graph(pd_notation):
    pd_list = ast.literal_eval(pd_notation)
    link = snappy.Link(pd_list)
    crossings = link.crossings

    edges = set()
    signs = []

    for c in crossings:
        signs.append(c.sign)
        for pos in range(4):
            neighbor, _ = c.adjacent[pos]
            edge = tuple(sorted((c.label, neighbor.label)))
            edges.add(edge)

    edges = list(edges)
    sources = np.array([e[0] for e in edges])
    targets = np.array([e[1] for e in edges])
    sign_indices = np.array([0 if s == -1 else 1 for s in signs])

    return sources, targets, sign_indices


class KnotDataset(InMemoryDataset):
    def __init__(self, root, csv_path, target='Signature', log_target=False, transform=None, pre_transform=None):
        self.csv_path = csv_path
        self.target = target
        self.log_target = log_target
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_path)]

    @property
    def processed_file_names(self):
        safe_name = self.target.lower().replace(' ', '_')
        suffix = '_log' if self.log_target else ''
        return [f'knot_graphs_{safe_name}{suffix}.pt']

    def process(self):
        knots_df = pd.read_csv(self.csv_path)
        data_list = []
        skipped = 0

        for idx in range(len(knots_df)):
            pd_notation = knots_df['PD Notation'].iloc[idx]
            target_val = knots_df[self.target].iloc[idx]

            if pd.isna(pd_notation) or pd.isna(target_val):
                skipped += 1
                continue

            try:
                pd_notation = fix_pd_notation(str(pd_notation))
                sources, targets, sign_indices = pd_to_crossing_graph(pd_notation)

                edge_index = torch.tensor(
                    np.stack([
                        np.concatenate([sources, targets]),
                        np.concatenate([targets, sources]),
                    ]),
                    dtype=torch.long,
                )
                x = torch.tensor(sign_indices, dtype=torch.long).unsqueeze(-1)
                val = float(target_val)
                if self.log_target:
                    val = math.log(val)
                y = torch.tensor(val, dtype=torch.float)

                data_list.append(Data(x=x, edge_index=edge_index, y=y))
            except Exception as e:
                if skipped < 3:
                    print(f"Row {idx} failed: {e}")
                skipped += 1
                continue

        print(f"Processed {len(data_list)} graphs, skipped {skipped}")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])


class KnotDataLoaders:
    def __init__(self, dataset, batch_size=64, split=(0.8, 0.1, 0.1), seed=42):
        n = len(dataset)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        n_test = n - n_train - n_val

        generator = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val = DataLoader(val_set, batch_size=batch_size)
        self.test = DataLoader(test_set, batch_size=batch_size)
