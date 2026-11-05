"""Configurable model registry.

Each entry knows how to build the network, which sub-module holds the classifier head (for
freezing/fine-tuning), which layer to use for Grad-CAM, and how to extract an embedding
(penultimate features) for OOD detection and similarity search.

Weights are only downloaded when a model is actually built with ``pretrained=True``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """Baseline CNN trained from scratch: stride-2 stem + 4 conv blocks (conv-BN-ReLU x2 + maxpool) + GAP + MLP head."""

    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w = width
        stem = nn.Sequential(nn.Conv2d(3, w, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.features = nn.Sequential(stem, block(w, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, 128),
                                        nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


@dataclass
class ArchSpec:
    name: str
    family: str
    builder: Callable[[int, bool, float], nn.Module]
    head_attr: str             # attribute path of the classification head
    cam_layer: str             # attribute path of the Grad-CAM target layer
    default_input: int = 224
    edge_friendly: bool = False
    notes: str = ""


def _tv(name, weights_enum):
    import torchvision.models as tvm

    def build(num_classes: int, pretrained: bool, dropout: float):
        weights = getattr(tvm, weights_enum).DEFAULT if pretrained else None
        m = getattr(tvm, name)(weights=weights)
        _replace_head(m, num_classes, dropout)
        return m
    return build


def _replace_head(m: nn.Module, num_classes: int, dropout: float):
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):  # resnet
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    elif hasattr(m, "classifier"):
        c = m.classifier
        if isinstance(c, nn.Linear):  # densenet
            m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(c.in_features, num_classes))
        else:  # mobilenet/efficientnet/convnext: last module is Linear
            last_idx = max(i for i, mod in enumerate(c) if isinstance(mod, nn.Linear))
            c[last_idx] = nn.Linear(c[last_idx].in_features, num_classes)
            for mod in c:
                if isinstance(mod, nn.Dropout):
                    mod.p = dropout
    else:
        raise ValueError("unknown head")


def _timm(name):
    def build(num_classes: int, pretrained: bool, dropout: float):
        import timm
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
    return build


ARCHITECTURES: dict[str, ArchSpec] = {
    "custom_cnn": ArchSpec("custom_cnn", "cnn", lambda n, p, d: CustomCNN(n, dropout=d), "classifier",
                           "features.4", 224, True, "Trained from scratch; baseline."),
    "mobilenet_v2": ArchSpec("mobilenet_v2", "cnn", _tv("mobilenet_v2", "MobileNet_V2_Weights"), "classifier",
                             "features.18", 224, True),
    "mobilenet_v3_small": ArchSpec("mobilenet_v3_small", "cnn", _tv("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
                                   "classifier", "features.12", 224, True, "Edge-deployment candidate."),
    "efficientnet_b0": ArchSpec("efficientnet_b0", "cnn", _tv("efficientnet_b0", "EfficientNet_B0_Weights"),
                                "classifier", "features.8", 224),
    "efficientnet_b3": ArchSpec("efficientnet_b3", "cnn", _tv("efficientnet_b3", "EfficientNet_B3_Weights"),
                                "classifier", "features.8", 300, notes="Heavier; trained only if resources permit."),
    "resnet50": ArchSpec("resnet50", "cnn", _tv("resnet50", "ResNet50_Weights"), "fc", "layer4", 224),
    "densenet121": ArchSpec("densenet121", "cnn", _tv("densenet121", "DenseNet121_Weights"), "classifier",
                            "features.denseblock4", 224),
    "convnext_tiny": ArchSpec("convnext_tiny", "cnn", _tv("convnext_tiny", "ConvNeXt_Tiny_Weights"), "classifier",
                              "features.7", 224),
    "vit_tiny": ArchSpec("vit_tiny", "transformer", _timm("vit_tiny_patch16_224.augreg_in21k_ft_in1k"), "head",
                         "blocks.11.norm1", 224, notes="ViT-Ti/16 (timm). ViT-B/16 is 330 MB; Ti is used for CPU budget."),
}


def get_module(model: nn.Module, path: str) -> nn.Module:
    m = model
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise KeyError(f"Unknown architecture '{arch}'. Available: {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch].builder(num_classes, pretrained, dropout)


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head (feature-extraction mode)."""
    head = get_module(model, ARCHITECTURES[arch].head_attr)
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable


class EmbeddingHook:
    """Captures the input to the final Linear layer = penultimate embedding."""

    def __init__(self, model: nn.Module, arch: str):
        head = get_module(model, ARCHITECTURES[arch].head_attr)
        linears = [m for m in head.modules() if isinstance(m, nn.Linear)] if not isinstance(head, nn.Linear) else [head]
        self.layer = linears[-1]
        self.value: torch.Tensor | None = None
        self.h = self.layer.register_forward_hook(lambda m, inp, out: setattr(self, "value", inp[0].detach()))

    def remove(self):
        self.h.remove()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
