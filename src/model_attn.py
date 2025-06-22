import timm
import torch
import torch.nn as nn

class ViTWithFeatures(nn.Module):
    def __init__(self, model_name, num_classes=9, selected_blocks=(3, 6, 9)):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.selected_blocks = selected_blocks

    def forward(self, x, return_features=False):
        features = []
        B = x.shape[0]

        x = self.model.patch_embed(x)
        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i in self.selected_blocks:
                features.append(x.clone())

        x = self.model.norm(x)
        cls_output = x[:, 0]  # CLS token output
        logits = self.model.head(cls_output)

        if return_features:
            return logits, features
        return logits

def get_vit_model(model_name, num_classes=9):
    return ViTWithFeatures(model_name, num_classes=num_classes)
