import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceMap(nn.Module):
    r"""DistanceMap
    """
    def __init__(self, num_bins, bin_displacement=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.bin_displacement = bin_displacement

    def forward(self, center, output_sz):
        """center: torch tensor with (y,x) center position
        output_sz: size of output"""

        bin_centers = torch.arange(self.num_bins, dtype=torch.float32, device=center.device).view(1, -1, 1, 1)

        k0 = torch.arange(output_sz[0], dtype=torch.float32, device=center.device).view(1,1,-1,1)
        k1 = torch.arange(output_sz[1], dtype=torch.float32, device=center.device).view(1,1,1,-1)

        d0 = k0 - center[:,0].view(-1,1,1,1)
        d1 = k1 - center[:,1].view(-1,1,1,1)

        dist = torch.sqrt(d0*d0 + d1*d1)
        bin_diff = dist / self.bin_displacement - bin_centers

        bin_val = torch.cat((F.relu(1.0 - torch.abs(bin_diff[:,:-1,:,:]), inplace=True),
                             (1.0 + bin_diff[:,-1:,:,:]).clamp(0, 1)), dim=1)

        return bin_val


