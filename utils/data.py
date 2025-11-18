from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

# Средние и стандартные отклонения каналов CIFAR-10 (нужны для нормализации)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def build_transforms(train: bool = True):
    """
    Возвращает пайплайн преобразований:
    - для train: аугментации + нормализация
    - для test: только нормализация
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])


def build_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2
):
    """
    Загружает CIFAR-10 и возвращает trainloader и testloader.
    """
    data_dir = Path(data_dir)

    trainset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=build_transforms(train=True),
    )

    testset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=build_transforms(train=False),
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, testloader


# Классы CIFAR-10
CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)
