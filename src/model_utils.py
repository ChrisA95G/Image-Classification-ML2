import torch
import torch.nn as nn
import timm
import torchmetrics
from src.config import NUM_CLASSES, FREEZE_BACKBONE, NUM_EPOCHS, PRETRAIND_MODEL

def create_model():
    """Creates the Vision Transformer model with a modified patch embedding for 4 channels."""
    model = timm.create_model(
        PRETRAIND_MODEL,
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )

    patch_embed_module = model.patch_embed
    original_conv_layer_candidate = None
    # layer_container_module will be the module whose attribute needs to be replaced
    # e.g., model.patch_embed or model.patch_embed.conv1
    layer_container_module = patch_embed_module 
    layer_name_to_replace = None

    # Try to find the first convolutional layer.
    # Common name: 'proj' (standard ViT)
    if hasattr(patch_embed_module, 'proj') and isinstance(patch_embed_module.proj, nn.Conv2d):
        original_conv_layer_candidate = patch_embed_module.proj
        layer_name_to_replace = 'proj'
    # Common name: 'conv1' (e.g., some custom ViTs or if PatchEmbed itself is a Conv2d wrapper)
    elif hasattr(patch_embed_module, 'conv1') and isinstance(patch_embed_module.conv1, nn.Conv2d):
        original_conv_layer_candidate = patch_embed_module.conv1
        layer_name_to_replace = 'conv1'
    # Nested structure: 'conv1.conv' (e.g., TinyViT where patch_embed.conv1 is a Conv2d_BN module)
    elif hasattr(patch_embed_module, 'conv1') and \
         hasattr(patch_embed_module.conv1, 'conv') and \
         isinstance(patch_embed_module.conv1.conv, nn.Conv2d):
        original_conv_layer_candidate = patch_embed_module.conv1.conv
        layer_container_module = patch_embed_module.conv1 # We need to replace 'conv' within 'conv1'
        layer_name_to_replace = 'conv'
    
    if original_conv_layer_candidate is None:
        raise AttributeError(
            f"Could not find a suitable Conv2D layer (tried 'proj', 'conv1', or 'conv1.conv' within model.patch_embed) "
            f"for model '{PRETRAIND_MODEL}'. Please check the model structure or adapt 'create_model'."
        )

    original_conv_layer = original_conv_layer_candidate
    original_weights = original_conv_layer.weight.data
    original_bias_data = original_conv_layer.bias.data if original_conv_layer.bias is not None else None

    new_conv_layer = nn.Conv2d(
        in_channels=4,
        out_channels=original_conv_layer.out_channels,
        kernel_size=original_conv_layer.kernel_size,
        stride=original_conv_layer.stride,
        padding=original_conv_layer.padding,
        dilation=original_conv_layer.dilation,
        groups=original_conv_layer.groups,
        bias=(original_bias_data is not None)
    )

    with torch.no_grad():
        new_weights_tensor = torch.zeros_like(new_conv_layer.weight.data)
        new_weights_tensor[:, 0, :, :] = original_weights[:, 1, :, :]  # Orig Green
        new_weights_tensor[:, 1, :, :] = original_weights[:, 2, :, :]  # Orig Blue
        new_weights_tensor[:, 2, :, :] = original_weights[:, 0, :, :]  # Orig Red
        new_weights_tensor[:, 3, :, :] = 0.5 * (original_weights[:, 0, :, :] + original_weights[:, 1, :, :]) # 0.5*(Orig R+G)
        new_conv_layer.weight.data = new_weights_tensor
        if original_bias_data is not None:
            new_conv_layer.bias.data = original_bias_data.clone()

    setattr(layer_container_module, layer_name_to_replace, new_conv_layer)
    return model


def setup_training(model):
    """Sets up the optimizer and learning rate scheduler."""
    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            param.requires_grad = any(
                [
                    name.startswith("head."),
                    name.startswith("norm."),
                    name.startswith("patch_embed."),
                    any(name.startswith(f"blocks.{i}.mlp.") for i in range(4, 8)),
                    any(name.startswith(f"blocks.{i}.norm") for i in range(4, 8)),
                    any(name.startswith(f"blocks.{i}.") for i in range(8, 12)),
                ]
            )

        param_groups = [
            {"params": [p for n, p in model.named_parameters() if n.startswith("head.")], "lr": 3e-4},
            {"params": [p for n, p in model.named_parameters() if n.startswith("patch_embed.")], "lr": 1e-4},
            {"params": [p for n, p in model.named_parameters() if any(n.startswith(f"blocks.{i}.") for i in range(8, 12))], "lr": 1e-5},
            {"params": [p for n, p in model.named_parameters() if any(n.startswith(f"blocks.{i}.mlp.") for i in range(4, 8)) or any(n.startswith(f"blocks.{i}.norm") for i in range(4, 8))], "lr": 5e-5},
        ]
        # Add any remaining unfrozen parameters with a default small LR
        other_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(p is pg_param for group in param_groups for pg_param in group['params'])]
        if other_params:
            param_groups.append({"params": other_params, "lr": 1e-5}) # Default LR for other unfrozen layers

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)

    warmup_epochs = 5
    cosine_epochs = NUM_EPOCHS - warmup_epochs
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, cosine])

    return optimizer, scheduler


def setup_metrics(device):
    """Initializes torchmetrics for training and validation."""
    metric_collection = lambda: torchmetrics.MetricCollection({
        "f1": torchmetrics.F1Score(task="multilabel", num_labels=NUM_CLASSES, average="macro"),
        "hamming": torchmetrics.HammingDistance(task="multilabel", num_labels=NUM_CLASSES),
        "exact_match": torchmetrics.ExactMatch(task="multilabel", num_labels=NUM_CLASSES),
    })
    metrics = {
        "train": metric_collection().to(device),
        "val": metric_collection().to(device),
    }
    return metrics