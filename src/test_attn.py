import argparse
import torch
from model_attn import get_vit_model
from dataloader import get_medmnist_loaders
from utils import evaluate  # Reuse your existing eval logic

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = get_vit_model(args.model, num_classes=9)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device)['model'])
    model.to(device)

    # Load test data only
    _, _, test_loader = get_medmnist_loaders(batch_size=args.batch_size, img_size=args.img_size)

    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT on MedMNIST test set")
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)

    args = parser.parse_args()
    main(args)
