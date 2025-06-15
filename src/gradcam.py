import numpy as np
from PIL import Image

def vit_gradcam(model, input_tensor, target_layer, class_idx=None):
    """
    Compute Grad-CAM for a ViT model.

    Args:
        model: ViT model from timm (in eval mode).
        input_tensor: Input tensor of shape (1, 3, 224, 224).
        target_layer: Layer of the model to hook for activations and gradients.
        class_idx: Target class index for Grad-CAM. If None, uses predicted class.

    Returns:
        cam: Grad-CAM heatmap resized to (224, 224), numpy uint8 array.
        class_idx: The class index used.
    """

    activations = []
    grads = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        grads.append(grad_output[0])

    # Register hooks
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax().item()

    score = output[0, class_idx]
    model.zero_grad()
    score.backward()

    # Remove hooks
    fh.remove()
    bh.remove()

    # Extract activations and grads
    act = activations[0][:, 1:, :]  # Remove CLS token (1, 196, C)
    grad = grads[0][:, 1:, :]

    # Compute weights and weighted sum (Grad-CAM)
    weights = grad.mean(dim=1, keepdim=True)  # (1, 1, C)
    cam = (weights * act).sum(dim=-1).squeeze()  # (196,)

    cam = cam.reshape(14, 14).detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize

    # Resize CAM to 224x224
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((224, 224), resample=Image.BILINEAR)
    cam_img = np.array(cam_img)

    return cam_img, class_idx
