# SeeWhatYouShrink

## Installation
install the required packages. You can do this using pip:

```bash
pip install -r requirements.txt
```
## Training from scratch
We train a VIT Base Model (Patch size 16, image size 224) and a VIT Tiny Model (Patch size 16, image size 224) on the MedMNIST dataset. To run the training, you can use the following command:

```bash
python scratch_trained/train.py --batch_size 64 --model vit_base_patch16_224 --epochs 30 --save_path vit_base_best.pth --lr 1e-5
```

To resume training from a checkpoint, you can use the following command:

```bash
python scratch_trained/train.py --batch_size 64 --model vit_base_patch16_224 --epochs 30 --save_path vit_base_best.pth --lr 1e-5 --resume
```

## Trained models
To download pretrained checkpoints, use the following command:

```bash
bash checkpoints/get_checkpoints.sh
```

Refer to [this folder](./checkpoints) for more details.

To evaluate the trained models, use the following command:

```bash
python scratch_trained/test.py --model <model_name> --ckpt_path <checkpoint_path>
```

The results from training are as follows:
| Model                  | Patch Size | Image Size | Test Accuracy (%) | Max Validation Accuracy (%) |
|------------------------|------------|------------|--------------|-----------------|
| ViT Base               | 16         | 224        | 83.16 | 98.03     |
| ViT Tiny               | 16         | 224        | 83.37 | 97.42     |

## GradCAM
To Do!

