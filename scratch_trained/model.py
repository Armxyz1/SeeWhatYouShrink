import timm

def get_vit_model(model_name, num_classes):
    return timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        in_chans=3
    )
