import torch
from torch.utils.data import Dataset
import cv2

# Using Albumentations for TPS-like transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class SingleImageDataset(Dataset):
    def __init__(self, image_path, augmentations_per_image=100):
        super(SingleImageDataset, self).__init__()
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Create an enhanced policy (TPS-like elastic transformation)
        self.transform = A.Compose([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            ToTensorV2()
        ])
        
        self.augmentations_per_image = augmentations_per_image

    def __len__(self):
        return self.augmentations_per_image

    def __getitem__(self, idx):
        # Original image (as output target)
        target_image = self.image.copy()

        # Random deformation augmentation (as input)
        augmented_image = self.transform(image=self.image)['image'] # uint8 Tensor

        # Normalization to [-1, 1]
        augmented_image = augmented_image.float() / 127.5 - 1.0

        # Convert to tensor, range [-1, 1]
        target_image = torch.from_numpy(target_image / 127.5 - 1.0).permute(2,0,1).float()

        # print("Aug dtype:", augmented_image.dtype)
        # print("Target dtype:", target_image.dtype)

        return augmented_image, target_image
