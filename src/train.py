import argparse
import torch
from model import get_vit_model
from dataloader import get_medmnist_loaders
from utils import *
import torch.optim as optim

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_vit_model(args.model, num_classes=9).to(device)
    train_loader, val_loader, _ = get_medmnist_loaders(args.batch_size, img_size=args.img_size)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        save_path=args.save_path or f"checkpoint_{args.model}.pth",
        resume=args.resume,
        patience=args.patience
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT models on PathMNIST")

    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224', help="ViT model name from timm")
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")


    args = parser.parse_args()
    main(args)
