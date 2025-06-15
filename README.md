# SeeWhatYouShrink

SeeWhatYouShrink is a project focused on efficient and interpretable deep learning for medical image classification. Leveraging Vision Transformers (ViT) and knowledge distillation, this repository demonstrates how to train, evaluate, and explain compact models on the MedMNIST dataset. The project includes scripts for model training, evaluation, GradCAM-based explainability, and knowledge distillation, along with pretrained checkpoints and visualizations.

## Installation

To install the required dependencies, please execute the following command:

```bash
pip install -r requirements.txt
```

## Training from Scratch

This project provides training scripts for both the ViT Base (Patch size 16, image size 224) and ViT Tiny (Patch size 16, image size 224) models on the MedMNIST dataset. To initiate training, use the command below:

```bash
python src/train.py --batch_size 64 --model vit_base_patch16_224 --epochs 30 --save_path vit_base_best.pth --lr 1e-5
```

To resume training from a previously saved checkpoint, use:

```bash
python src/train.py --batch_size 64 --model vit_base_patch16_224 --epochs 30 --save_path vit_base_best.pth --lr 1e-5 --resume
```

## Pretrained Models

Pretrained model checkpoints can be downloaded using the following script:

```bash
bash checkpoints/get_checkpoints.sh
```

For additional details, please refer to the [checkpoints directory](./checkpoints).

To evaluate a trained model, execute:

```bash
python src/test.py --model <model_name> --ckpt_path <checkpoint_path>
```

The performance metrics for the trained models are summarized below:

| Model      | Test Accuracy (%) | Max Validation Accuracy (%) |
|------------|------------------|----------------------------|
| ViT Base   | 83.16            | 98.03                      |
| ViT Tiny   | 83.37            | 97.42                      |

## GradCAM Explainability

To conduct explainability experiments using GradCAM, run:

```bash
python3 src/explainability.py --model <model_name> --ckpt_path <checkpoint_path> --save_path <save_path>
```

### Overview of GradCAM

GradCAM (Gradient-weighted Class Activation Mapping) highlights the regions of an input image that most significantly influence the model’s predictions. It operates by computing the gradients of the target class with respect to the patch embeddings from a transformer block (typically the final one), excluding the [CLS] token. These gradients are globally averaged to obtain weights, which are then used to compute a weighted sum over the activations. The resulting 14×14 attention map is upscaled to 224×224 to match the input image size, producing a heatmap that visually indicates the areas of focus during inference. Regions with higher gradient magnitudes are considered more influential in the model’s decision-making process.

### Visualizations

**ViT Base Model:**

![ViT Base GradCAM Example](gradcam_plots/gradcam_overlay_base.png)

**ViT Tiny Model:**

![ViT Tiny GradCAM Example](gradcam_plots/gradcam_overlay_tiny.png)

Both models demonstrate strong predictive performance; however, the ViT Base model offers more interpretable visualizations, providing clearer insights into the regions influencing its predictions.

## Knowledge Distillation

Knowledge distillation is a model compression technique in which a smaller, more efficient "student" model is trained to replicate the behavior of a larger "teacher" model. The student learns from both the ground truth labels and the soft predictions of the teacher, enabling it to achieve competitive performance with reduced computational requirements. This approach is particularly beneficial for deploying models on resource-constrained devices, as it preserves much of the teacher’s expressiveness while improving efficiency.

To perform knowledge distillation, use the following command:

```bash
python src/kd.py --teacher_model <teacher_model_name> --student_model <student_model_name> --ckpt_path <checkpoint_path> --save_path <save_path> --epochs 30 --batch_size 64 --lr 1e-5
```

Pretrained knowledge distillation checkpoints are available in the [checkpoints directory](./checkpoints).

The results obtained from knowledge distillation are as follows:

| Model        | Test Accuracy (%) | Max Validation Accuracy (%) |
|--------------|------------------|----------------------------|
| ViT KD Tiny  | 83.20            | 98.34                      |

### GradCAM for Knowledge Distillation

GradCAM experiments conducted on the knowledge-distilled student model indicate that it effectively learns the expressiveness of the teacher model. The resulting heatmaps demonstrate that the student model’s decision-making process closely mirrors that of the teacher, while maintaining efficiency in terms of size and inference time.

![ViT KD Tiny GradCAM Example](gradcam_plots/gradcam_overlay_kd_tiny.png)

## Final Results

The following table summarizes the performance and efficiency metrics for all evaluated models:

| Model        | Inference Time per Sample (s) | Number of Parameters | Test Accuracy (%) |
|--------------|-------------------------------|----------------|-------------------|
| ViT Base     | 9.12                          | 83.8M          | 83.16             |
| ViT Tiny     | 0.86                          | 5.4M           | 83.37             |
| ViT KD Tiny  | 0.87                          | 5.4M           | 83.20             |

