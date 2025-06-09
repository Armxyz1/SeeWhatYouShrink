from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import INFO

def get_medmnist_loaders(batch_size=64, img_size=64):
    info = INFO['pathmnist']
    DataClass = getattr(__import__('medmnist'), info['python_class'])

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_ds = DataClass(split='train', transform=transform, download=True)
    val_ds = DataClass(split='val', transform=transform, download=True)
    test_ds = DataClass(split='test', transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
