import os

import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


# Функция для вычисления IoU
def calculate_iou(preds, labels, smooth=1e-6):
    preds = torch.sigmoid(preds) > 0.5  # бинаризация
    intersection = torch.logical_and(preds, labels).float().sum((1, 2))
    union = torch.logical_or(preds, labels).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def save_augmented_images(images, masks, epoch, save_dir="augmented_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (image, mask) in enumerate(zip(images, masks)):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(image_np)
        ax[0].set_title("Augmented Image")
        ax[0].axis("off")

        ax[1].imshow(mask_np, cmap='gray')
        ax[1].set_title("Augmented Mask")
        ax[1].axis("off")

        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_sample_{idx}.png"))
        plt.close()


def save_predictions(model, val_loader, device, save_dir, epoch):
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)  # No ['out'] needed

            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.float()

            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()

            if inputs_np.shape[0] > 0:
                visualize_segmentation(inputs_np, targets_np, preds_np, save_dir, epoch, 0)


def visualize_segmentation(images_np, masks_np, preds_np, save_dir, epoch, idx):
    num_images = min(images_np.shape[0], 3)

    fig, ax = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    if num_images == 1:
        ax = [ax]

    for i in range(num_images):
        ax[i, 0].imshow(images_np[i].transpose(1, 2, 0))  # Входное изображение
        ax[i, 0].set_title("Input Image")
        ax[i, 0].axis('off')

        ax[i, 1].imshow(masks_np[i].squeeze(), cmap='gray')  # Маска
        ax[i, 1].set_title("Ground Truth")
        ax[i, 1].axis('off')

        ax[i, 2].imshow(preds_np[i].squeeze(), cmap='gray')  # Предсказание
        ax[i, 2].set_title("Prediction")
        ax[i, 2].axis('off')

    plt.savefig(f"{save_dir}/epoch_{epoch}_batch_{idx}.png")
    plt.close()
