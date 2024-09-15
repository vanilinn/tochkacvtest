import os
import cv2
import torch
import numpy as np
from torchvision import transforms

from model import get_model


def load_model(model_path, device):
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image, device, input_size=(512, 512)):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    return input_tensor


# Процесс инференса для одной картинки
def infer_model(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)
        pred_mask = torch.sigmoid(prediction).cpu().numpy()
    return pred_mask[0, 0]  # Одноканальная (бинарная) маска


# Объединение масок в одну
def combine_masks(masks):
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)  # Используем максимумы для объединения
    return combined_mask


def find_optimal_circle(mask):
    # Преобразование маски в формат, пригодный для поиска контуров
    mask = (mask * 255).astype(np.uint8)  # Преобразование в 8-битный формат

    # Убираем шум с помощью морфологических операций (например, открытие)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    # Оставляем только контуры, которые не касаются границ изображения
    height, width = mask.shape
    valid_contours = []
    for contour in contours:
        if not np.any(contour == 0) and not np.any(contour[:, :, 0] == width - 1) and not np.any(
                contour[:, :, 1] == height - 1):
            valid_contours.append(contour)

    if len(valid_contours) == 0:
        return None, None
    print(valid_contours)

    # Получение минимальной окружности, которая охватывает все валидные контуры
    all_points = np.vstack(valid_contours)  # Собираем все точки всех валидных контуров
    (x, y), radius = cv2.minEnclosingCircle(all_points)
    # Отладочная информация
    print(f"Найдена окружность с центром в ({x}, {y}) и радиусом {radius}")
    return (x, y), radius


# Обработка изображений в папке
def process_images_in_folder(model, input_folder, output_folder, device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)

        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Оригинальный размер изображения
        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Преобразование изображения для модели
        image_tensor = preprocess_image(image, device)

        # Предсказание маски (если модель предсказывает несколько масок, соберем их)
        mask = infer_model(model, image_tensor)

        # Преобразование предсказанной маски к оригинальному размеру изображения
        resized_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)

        # Если маска предсказывается несколькими частями, объединяем их
        combined_mask = combine_masks([resized_mask])

        # Поиск оптимальной окружности по объединенной маске
        center, radius = find_optimal_circle(combined_mask)

        # Отрисовка маски и окружности на изображении
        result_image = draw_mask_and_circle_on_image(image, combined_mask, center, radius)

        # Сохранение результата
        output_image_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_image_path, result_image)


# Отрисовка маски и окружности на изображении
def draw_mask_and_circle_on_image(image, mask, center, radius):
    # Преобразуем изображение в черно-белое
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colored_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Создаем красную маску для overlay
    red_overlay = np.zeros_like(colored_image)
    red_overlay[mask > 0.5] = [0, 0, 255]  # Красный цвет для маски

    # Объединение изображений с прозрачностью
    combined_image = cv2.addWeighted(colored_image, 0.8, red_overlay, 0.2, 0)

    # Рисуем окружность и центр
    if center is not None:
        cv2.circle(combined_image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)  # Центр окружности
        if radius is not None:
            cv2.circle(combined_image, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)

    return combined_image


if __name__ == "__main__":
    device = torch.device('cpu')  # cpu из-за того, что gpu занят в этот момент обучением
    input_directory = "00_data/test_real"  # Папка с исходными изображениями
    output_directory = "results_circle1"  # Папка для сохранения результатов
    model_path = "model_epoch_14.pth"  # Путь к предобученной модели

    # Load the model
    model = load_model(model_path, device)

    # Process images in the folder
    process_images_in_folder(model, input_directory, output_directory, device)
