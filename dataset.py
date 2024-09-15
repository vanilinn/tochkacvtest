import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import cv2


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False, num_augmentations=2):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.num_augmentations = num_augmentations

        self.transform = A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths) * self.num_augmentations

    def __getitem__(self, idx):
        img_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations

        image = cv2.imread(self.image_paths[img_idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)

        if self.augment:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image'].float() / 255.0
            mask = augmented['mask'].float().unsqueeze(0) / 255.0
        else:
            image = cv2.resize(image, (512, 512))
            mask = cv2.resize(mask, (512, 512))
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return image, mask


