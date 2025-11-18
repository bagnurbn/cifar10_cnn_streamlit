import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarCNN(nn.Module):
    """
    Простая CNN для CIFAR-10:
    - несколько сверточных слоёв + ReLU + MaxPool
    - в конце полносвязные слои
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Блоки "фичей" (свертки + активации + pooling)
        self.features = nn.Sequential(
            # Блок 1: 3 канала (RGB) -> 64 каналов, размер остаётся 32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # уменьшаем размер: 32x32 -> 16x16

            # Блок 2: 64 -> 128 каналов, 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 16x16 -> 8x8

            # Блок 3: 128 -> 256 каналов, 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )

        # Классификатор (линейные слои)
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 256 * 4 * 4 -> в длинный вектор
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),  # выход: 10 классов
        )

    def forward(self, x):
        x = self.features(x)     # прогоняем через сверточные слои
        x = self.classifier(x)   # и через полносвязные
        return x
