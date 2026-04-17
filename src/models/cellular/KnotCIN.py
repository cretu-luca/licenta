import torch
import torch.nn.functional as F
import torch.nn as nn

from topomodelx.nn.cell.cwn import CWN


class KnotCIN(nn.Module):
    def __init__(self, in_channels_0, in_channels_1, in_channels_2, hid_channels=16, num_classes=1, n_layers=2):
        super().__init__()
        
        self.base_model = CWN(
            in_channels_0,
            in_channels_1,
            in_channels_2,
            hid_channels=hid_channels,
            n_layers=n_layers,
        )

        self.lin_0 = torch.nn.Linear(hid_channels, num_classes)
        self.lin_1 = torch.nn.Linear(hid_channels, num_classes)
        self.lin_2 = torch.nn.Linear(hid_channels, num_classes)

    def forward(self, x_0, x_1, x_2, adjacency_1, incidence_2, incidence_1_t):
        x_0, x_1, x_2 = self.base_model(
            x_0, x_1, x_2, adjacency_1, incidence_2, incidence_1_t
        )

        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)

        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0

        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0

        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0

        return (
            two_dimensional_cells_mean
            + one_dimensional_cells_mean
            + zero_dimensional_cells_mean
        )