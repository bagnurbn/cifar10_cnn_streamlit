import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from models.cifar_cnn import CifarCNN
from utils.data import build_dataloaders, CLASSES


def accuracy(logits, targets):
    """Простая функция для подсчёта accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def evaluate(model, loader, device):
    """Оценка модели на валидации."""
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            b = y.size(0)
            loss_sum += loss.item() * b
            acc_sum += accuracy(logits, y) * b
            n += b

    return loss_sum / n, acc_sum / n


def train_model():
    # Выбираем девайс (GPU если есть)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Загружаем CIFAR-10
    trainloader, testloader = build_dataloaders(batch_size=128, num_workers=2)

    # Создаём модель
    model = CifarCNN(num_classes=len(CLASSES)).to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Scheduler для плавного уменьшения LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_acc = 0.0
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    save_path = artifacts / "cifar_net.pth"

    epochs = 10

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()

        pbar = tqdm(trainloader)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"loss={loss.item():.3f}")

        scheduler.step()

        val_loss, val_acc = evaluate(model, testloader, device)
        print(f"Val loss={val_loss:.4f}  acc={val_acc:.4f}")

        # сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict()}, save_path)
            print(f"Saved best model with acc={best_acc:.4f}")

    print("\nTraining completed!")
    print("Final best accuracy:", best_acc)


if __name__ == "__main__":
    train_model()
