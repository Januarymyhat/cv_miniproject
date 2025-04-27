import numpy as np
import torch

# NeRF positional encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_freqs=10):
        super(PositionalEncoding, self).__init__()
        self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)

    def forward(self, coords):
        # coords: [B, 2, H, W], pixel coordinate information (x, y)
        encoding = [coords]
        for freq in self.freq_bands:
            encoding.append(torch.sin(freq * coords * np.pi))
            encoding.append(torch.cos(freq * coords * np.pi))
        return torch.cat(encoding, dim=1)  # Output high-dimensional features [B, C, H, W]
