# test.py
import torch
from torchvision.utils import save_image
import os
from models.generator import UNetGenerator
from dataset.single_image_dataset import SingleImageDataset
from torch.utils.data import DataLoader

# weights_path = 'results/with_positional_encoding/netG_final.pth'
weights_path = 'results/without_positional_encoding/netG_final.pth'
image_path = 'dataset/jcsmr.jpg'
save_dir = 'test_results'
os.makedirs(save_dir, exist_ok=True)

# Model and device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_nc = 3  # Number of input channels (not using positional encoding)

netG = UNetGenerator(input_nc=input_nc, output_nc=3).to(device)
netG.load_state_dict(torch.load(weights_path, map_location=device))
netG.eval()

# Data preparation
# Use the same data augmentation method as training (only generate 1 image)
dataset = SingleImageDataset(image_path=image_path, augmentations_per_image=1)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Test Generation
with torch.no_grad():
    for i, (input_img, _) in enumerate(dataloader):
        input_img = input_img.to(device)
        fake_img = netG(input_img)
        # save_image(fake_img * 0.5 + 0.5, os.path.join(save_dir, f'fake_test_output_with.png'))
        # save_image(input_img * 0.5 + 0.5, os.path.join(save_dir, f'input_test_image_with.png'))
        save_image(fake_img * 0.5 + 0.5, os.path.join(save_dir, f'fake_test_output_without.png'))
        save_image(input_img * 0.5 + 0.5, os.path.join(save_dir, f'input_test_image_without.png'))
        print("Generated image saved to test_results/")
        break  # Save only one
