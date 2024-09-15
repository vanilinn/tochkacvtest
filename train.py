import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model import get_model, WeightedIOULoss, DiceLoss, CombinedLoss, FocalLoss
from dataset import SegmentationDataset
from utils import calculate_iou, save_predictions, save_augmented_images
import matplotlib.pyplot as plt
import time

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, loss_fn, train_loader, val_loader, num_epochs=50, learning_rate=1e-4,
                save_dir="models_augmented"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses, val_losses, ious = [], [], []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Сохраняем аугментированные изображения после каждой эпохи
        save_augmented_images(inputs.cpu(), targets.cpu(), epoch)

        train_loss = running_loss / len(train_loader)
        val_loss, val_iou = validate_model(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        ious.append(val_iou)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Time: {epoch_time:.2f} seconds")

        torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch + 1}.pth")

        save_predictions(model, val_loader, device, save_dir, epoch)
        scheduler.step(val_loss)

    return train_losses, val_losses, ious


def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    IoU_score = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
            IoU_score += calculate_iou(outputs, targets)

    return val_loss / len(val_loader), IoU_score / len(val_loader)


# Функция для чтения списка файлов
def read_file_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


if __name__ == "__main__":
    data_root = "01_generate_dataset/pipe_voc_dataset/pipe_voc_data"

    image_dir = os.path.join(data_root, "JPEGImages")
    mask_dir = os.path.join(data_root, "SegmentationClass")

    train_list = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    val_list = os.path.join(data_root, "ImageSets", "Segmentation", "val.txt")

    train_files = read_file_list(train_list)
    val_files = read_file_list(val_list)

    train_image_paths = [os.path.join(image_dir, f"{file_id}.jpg") for file_id in train_files]
    train_mask_paths = [os.path.join(mask_dir, f"{file_id}.png") for file_id in train_files]

    val_image_paths = [os.path.join(image_dir, f"{file_id}.jpg") for file_id in val_files]
    val_mask_paths = [os.path.join(mask_dir, f"{file_id}.png") for file_id in val_files]

    train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, augment=True)
    val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = get_model(pretrained=True)
    # model.load_state_dict(torch.load('models_augmented_vis_10/model_epoch_1.pth'))

    loss_fn = WeightedIOULoss().to(device)

    train_losses, val_losses, ious = train_model(model, loss_fn, train_loader, val_loader, num_epochs=50)
    train_losses = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_losses]
    val_losses = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in val_losses]
    ious = [iou.cpu().item() if torch.is_tensor(iou) else iou for iou in ious]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(ious, label='Validation IoU')
    plt.legend()
    plt.title('IoU')

    plt.show()
