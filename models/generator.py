import torch
import torch.nn as nn

# Pix2Pix U-Net Architecture
class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=7, ngf=64):
        super(UNetGenerator, self).__init__()

        # Encoding layer (downsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # Decoding layer (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
