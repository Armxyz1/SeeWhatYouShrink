import argparse
import torch
from model_feat import get_vit_model
from dataloader import get_medmnist_loaders
from distill_utils_2 import *
import torch.optim as optim

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    projector = FeatureProjector(in_dim=192, out_dim=768).to(device)
    student_model = get_vit_model(args.student_model, num_classes=9).to(device)
    teacher_model = get_vit_model(args.teacher_model, num_classes=9).to(device)

    # Load teacher checkpoint if provided
    if args.teacher_ckpt:
        checkpoint = torch.load(args.teacher_ckpt, map_location=device)
        teacher_model.model.load_state_dict(checkpoint['model'])
        print(f"Loaded teacher model weights from {args.teacher_ckpt}")

    train_loader, val_loader, _ = get_medmnist_loaders(args.batch_size, img_size=args.img_size)

    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    run_training_distillation(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        projector=projector,  # ← here
        epochs=args.epochs,
        save_path=args.save_path or f"checkpoint_{args.student_model}.pth",
        resume=args.resume,
        patience=args.patience
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT models with knowledge distillation on PathMNIST")

    parser.add_argument('--student_model', type=str, default='vit_tiny_patch16_224', help="Student ViT model name from timm")
    parser.add_argument('--teacher_model', type=str, default='vit_base_patch16_224', help="Teacher ViT model name from timm")
    parser.add_argument('--teacher_ckpt', type=str, default='checkpoints/vit_base_best.pth', help="Path to teacher model checkpoint")
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")

    args = parser.parse_args()
    main(args)
