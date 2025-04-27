import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(PatchDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)
