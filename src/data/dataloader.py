"""DataLoader construction for the WikiArt multi-attribute dataset.

Provides ``build_dataloaders`` — a single entry point that wires together
:class:`~src.data.dataset.WikiArtDataset`, the transform pipelines, and an
optional :class:`~torch.utils.data.WeightedRandomSampler` to counter class
imbalance during training.

Also exports ``collate_fn`` which stacks image tensors and merges the list
of per-sample label dicts into a single dict of batched tensors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import WikiArtDataset
from src.data.transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Custom collate
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, int]]],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Collate a list of (image, labels_dict) samples into a batch.

    Args:
        batch: List of ``(image_tensor, labels_dict)`` tuples as returned by
            :class:`~src.data.dataset.WikiArtDataset.__getitem__`.

    Returns:
        A 2-tuple ``(images, labels)`` where:

        - **images** — stacked tensor of shape ``(B, C, H, W)``.
        - **labels** — dict mapping attribute name to a ``(B,)`` int64 tensor.
    """
    images, label_dicts = zip(*batch)
    image_batch = torch.stack(images, dim=0)  # (B, C, H, W)

    # Merge list of dicts → dict of tensors
    attributes = list(label_dicts[0].keys())
    label_batch: Dict[str, torch.Tensor] = {
        attr: torch.tensor([d[attr] for d in label_dicts], dtype=torch.long)
        for attr in attributes
    }
    return image_batch, label_batch


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_dataloaders(
    image_dir: str,
    style_train_csv: str,
    style_val_csv: str,
    genre_train_csv: str,
    genre_val_csv: str,
    artist_train_csv: str,
    artist_val_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
    sampler_attribute: str = "style",
) -> Tuple[DataLoader, DataLoader]:
    """Construct train and validation DataLoaders for WikiArt.

    Args:
        image_dir: Root directory containing WikiArt images.
        style_train_csv: Path to style training CSV.
        style_val_csv: Path to style validation CSV.
        genre_train_csv: Path to genre training CSV.
        genre_val_csv: Path to genre validation CSV.
        artist_train_csv: Path to artist training CSV.
        artist_val_csv: Path to artist validation CSV.
        batch_size: Number of samples per batch (default 32).
        num_workers: Subprocess workers for data loading (default 4).
        pin_memory: Pin CPU tensors to GPU memory for faster transfers
            (default ``True``).
        use_weighted_sampler: If ``True``, wrap the training dataset in a
            :class:`WeightedRandomSampler` to counteract class imbalance
            (default ``True``).
        sampler_attribute: Which attribute's class distribution to use for
            computing sample weights (default ``"style"``).

    Returns:
        A 2-tuple ``(train_loader, val_loader)``.
    """
    train_csv_files = {
        "style": style_train_csv,
        "genre": genre_train_csv,
        "artist": artist_train_csv,
    }
    val_csv_files = {
        "style": style_val_csv,
        "genre": genre_val_csv,
        "artist": artist_val_csv,
    }

    train_dataset = WikiArtDataset(
        image_dir=image_dir,
        csv_files=train_csv_files,
        transform=get_train_transforms(),
    )
    val_dataset = WikiArtDataset(
        image_dir=image_dir,
        csv_files=val_csv_files,
        transform=get_val_transforms(),
    )

    # Build sampler for training
    train_sampler: Optional[WeightedRandomSampler] = None
    shuffle_train = True
    if use_weighted_sampler:
        weights = train_dataset.get_sample_weights(sampler_attribute)
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle_train = False  # sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader
