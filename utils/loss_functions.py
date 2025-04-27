import torch
import torch.nn as nn

# GAN损失函数 (Least Squares GAN)
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)

# 像素重建损失（L1 loss）
L1Loss = nn.L1Loss()
