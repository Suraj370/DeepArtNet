"""Image preprocessing pipelines for WikiArt training and validation.

Provides two torchvision transform pipelines:
  - ``get_train_transforms`` — augmented pipeline for training
  - ``get_val_transforms``   — deterministic pipeline for validation/inference

Both pipelines normalise with ImageNet mean/std so they are compatible with
the pretrained EfficientNet-B4 backbone.
"""

from __future__ import annotations

from torchvision import transforms

# ImageNet statistics used by timm / EfficientNet-B4
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Return the augmented training transform pipeline.

    Pipeline::

        Resize(256)
        → RandomResizedCrop(image_size, scale=(0.7, 1.0))
        → RandomHorizontalFlip(p=0.5)
        → ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        → RandomRotation(15°)
        → RandomGrayscale(p=0.05)
        → ToTensor
        → Normalize(ImageNet mean/std)

    Args:
        image_size: Final crop size (height and width).  Default 224.

    Returns:
        A ``torchvision.transforms.Compose`` object ready to be passed to
        :class:`~src.data.dataset.WikiArtDataset`.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Return the deterministic validation / inference transform pipeline.

    Pipeline::

        Resize(256)
        → CenterCrop(image_size)
        → ToTensor
        → Normalize(ImageNet mean/std)

    Args:
        image_size: Final crop size (height and width).  Default 224.

    Returns:
        A ``torchvision.transforms.Compose`` object ready to be passed to
        :class:`~src.data.dataset.WikiArtDataset`.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
