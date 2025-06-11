import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from medmnist import PathMNIST
from gradcam import vit_gradcam
from model import get_vit_model
import argparse

def main(args):
    transform_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    dataset = PathMNIST(split='test', download=True)
    img, label = dataset[4]  # Load test image
    input_tensor = transform_preprocess(img).unsqueeze(0)  # (1, 3, 224, 224)

    model = get_vit_model(args.model, num_classes=9)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')

    model.load_state_dict(checkpoint['model']) 
    model.eval()

    target_layer = model.blocks[-1].norm1

    cam, class_idx = vit_gradcam(model, input_tensor, target_layer)

    img_np = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = img_np * 0.5 + 0.5  # Assuming normalization mean=0.5, std=0.5
    img_np = np.clip(img_np, 0, 1)

    # Convert CAM (uint8 0-255) to heatmap RGB [0-1]
    cam_heatmap = plt.get_cmap('jet')(cam / 255.0)[..., :3]

    # Overlay heatmap on original image
    overlay = 0.5 * img_np + 0.5 * cam_heatmap
    overlay = np.clip(overlay, 0, 1)

    # Plot side by side
    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img_np)
    axs[0].set_title(f"Original Image\nOriginal Class = {label[0]}")
    axs[0].axis('off')

    axs[1].imshow(overlay)
    axs[1].set_title(f"Grad-CAM Overlay\nPredicted Class = {class_idx}")
    axs[1].axis('off')

    plt.savefig(args.save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradCAM on VIT model")

    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224', help="ViT model name from timm")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default="gradcam_plots/gradcam_overlay_tiny.png")

    args = parser.parse_args()
    main(args)
