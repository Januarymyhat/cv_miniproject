import torch
from torch.utils.data import DataLoader
from dataset.single_image_dataset import SingleImageDataset
from models.positional_encoding import PositionalEncoding
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from utils.loss_functions import GANLoss, L1Loss
import torchvision.utils as vutils
import os


# Make sure the results folder exists
os.makedirs("results", exist_ok=True)

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Training parameters
num_epochs = 10
batch_size = 1
image_path = 'dataset/jcsmr.jpg'

# Loading dataset
dataset = SingleImageDataset(image_path=image_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
pos_encoder = PositionalEncoding().to(device)
netG = UNetGenerator(input_nc=3, output_nc=3).to(device)  # Input 3 channels (without encoding)
netD = PatchDiscriminator(input_nc=6).to(device)          # The input is the concatenation of input_img + target_img = 6 channels in total

# Loss function and optimizer
criterionGAN = GANLoss().to(device)
criterionL1 = L1Loss.to(device)

optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Training
for epoch in range(num_epochs):
    for i, (input_img, target_img) in enumerate(dataloader):
        input_img = input_img.to(device)       # [B, 3, H, W]
        target_img = target_img.to(device)     # [B, 3, H, W]

        # =============== Train Discriminator
        optimizerD.zero_grad()

        # Real image pairs
        real_pair = torch.cat((input_img, target_img), dim=1)  # [B, 6, H, W]
        pred_real = netD(real_pair)
        loss_D_real = criterionGAN(pred_real, True)

        # Generate image pairs
        fake_img = netG(input_img)
        fake_pair = torch.cat((input_img, fake_img.detach()), dim=1)  # detach prevents updating generators
        pred_fake = netD(fake_pair)
        loss_D_fake = criterionGAN(pred_fake, False)

        # Discriminator total loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizerD.step()

        # =============== Train Generator
        optimizerG.zero_grad()

        fake_img = netG(input_img)
        fake_pair = torch.cat((input_img, fake_img), dim=1)
        pred_fake = netD(fake_pair)

        # Generator loss = GAN loss + L1 reconstruction loss
        loss_G_GAN = criterionGAN(pred_fake, True)
        loss_G_L1 = criterionL1(fake_img, target_img) * 100.0  # L1加权
        loss_G = loss_G_GAN + loss_G_L1

        loss_G.backward()
        optimizerG.step()

        # Logging
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                  f"D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}, L1: {loss_G_L1.item():.4f}")

        # Save the generated image
        if i == 0:
            # vutils.save_image(input_img * 0.5 + 0.5, f"results/input_epoch_{epoch}.png")  # I forgot to save input img, don't have time to re-train ;(
            vutils.save_image((fake_img + 1) * 0.5, f"results/fake_epoch_{epoch}.png")
            vutils.save_image((target_img + 1) * 0.5, f"results/real_epoch_{epoch}.png") # same img

    # Save the model each time
    torch.save(netG.state_dict(), f"results/netG_epoch_{epoch}.pth")
    torch.save(netD.state_dict(), f"results/netD_epoch_{epoch}.pth")

# Save the final model
torch.save(netG.state_dict(), "results/netG_final.pth")
torch.save(netD.state_dict(), "results/netD_final.pth")
print("Training finished. Models saved.")
