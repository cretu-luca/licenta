import ast
import os
import numpy as np
import torch
import pandas as pd
import snappy
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from config import TASKS


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


def _parse_target(raw_val, task_cfg):
    if task_cfg.get('type') == 'binary':
        if raw_val == 'Y':
            return 1
        elif raw_val == 'N':
            return 0
        else:
            return None
    val = int(float(raw_val))
    class_idx = val - task_cfg['min_val']
    if class_idx < 0 or class_idx >= task_cfg['num_classes']:
        return None
    return class_idx


class KnotDataset(InMemoryDataset):
    def __init__(self, root, csv_path, task_name='Signature', transform=None, pre_transform=None):
        self.csv_path = csv_path
        self.task_name = task_name
        self.task_cfg = TASKS[task_name]
        self.n_classes = self.task_cfg['num_classes']
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_path)]

    @property
    def processed_file_names(self):
        safe_name = self.task_name.lower().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
        return [f'knot_cls_{safe_name}.pt']

    def process(self):
        knots_df = pd.read_csv(self.csv_path)
        col = self.task_cfg['col']
        data_list = []
        skipped = 0

        for idx in range(len(knots_df)):
            pd_notation = knots_df['PD Notation'].iloc[idx]
            raw_target = knots_df[col].iloc[idx]

            if pd.isna(pd_notation) or pd.isna(raw_target):
                skipped += 1
                continue

            try:
                class_idx = _parse_target(raw_target, self.task_cfg)
                if class_idx is None:
                    skipped += 1
                    continue

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
                y = torch.tensor(class_idx, dtype=torch.long)

                data_list.append(Data(x=x, edge_index=edge_index, y=y))
            except Exception as e:
                if skipped < 3:
                    print(f"Row {idx} failed: {e}")
                skipped += 1
                continue

        print(f"[{self.task_name}] Processed {len(data_list)} graphs, skipped {skipped}")

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
