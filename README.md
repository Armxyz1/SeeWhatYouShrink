# SeeWhatYouShrink

## Installation
install the required packages. You can do this using pip:

```bash
pip install -r requirements.txt
```
## Training from scratch
We train a VIT Base Model (Patch size 16, image size 224) and a VIT Tiny Model (Patch size 16, image size 224) on the MedMNIST dataset. To run the training, you can use the following command:

```bash
python src/train.py --batch_size 64 --model vit_base_patch16_224 --epochs 30 --save_path vit_base_best.pth --lr 1e-5
```

To resume training from a checkpoint, you can use the following command:

```bash
python src/train.py --batch_size 64 --model vit_base_patch16_224 --epochs 30 --save_path vit_base_best.pth --lr 1e-5 --resume
```

## Trained models
To download pretrained checkpoints, use the following command:

```bash
bash checkpoints/get_checkpoints.sh
```

Refer to [this folder](./checkpoints) for more details.

To evaluate the trained models, use the following command:

```bash
python src/test.py --model <model_name> --ckpt_path <checkpoint_path>
```

The results from training are as follows:
| Model                  | Patch Size | Image Size | Test Accuracy (%) | Max Validation Accuracy (%) |
|------------------------|------------|------------|--------------|-----------------|
| ViT Base               | 16         | 224        | 83.16 | 98.03     |
| ViT Tiny               | 16         | 224        | 83.37 | 97.42     |

## GradCAM
To run explainability experiment using GradCAM, run the following command:

```bash
python3 src/explainability.py --model <model_name> --ckpt_path <checkpoint_path> --save_path <save_path>
```

### What does a GradCAM do?
GradCAM highlights the image regions that most influenced the model’s prediction. It works by computing the gradients of the target class with respect to the patch embeddings from a transformer block (usually the last one), excluding the [CLS] token. These gradients are globally averaged to obtain weights, which are then used to compute a weighted sum over the activations. This produces a 14×14 attention map, which is upscaled to 224×224 to match the input image size. The resulting heatmap visually indicates where the model "focused" when making its decision—the higher the gradient magnitude, the greater the region's importance.

### Visualisations
Base model:
![ViT Base GradCAM Example](gradcam_plots/gradcam_overlay_base.png)

Tiny Model
![ViT Tiny GradCAM Example](gradcam_plots/gradcam_overlay_tiny.png)

While both models are capable of accurately predicting the correct class, the base model demonstrates greater explainability compared to the tiny model. Specifically, the base model provides more insightful visualizations of the regions it focuses on during prediction, making it easier to interpret and understand the decision-making process of the model.

