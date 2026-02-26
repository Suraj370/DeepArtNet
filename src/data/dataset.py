"""WikiArt dataset loader for multi-attribute art classification.

Reads pre-split CSV files for style, genre, and artist labels, builds a
union of all image paths, and returns ``(image_tensor, labels_dict)`` pairs.
Missing labels (an image present in one CSV but not another) are represented
by ``fallback_label`` (default -1) so the loss function can mask them out.

CSV format (one row per image)::

    StyleFolder/artist_painting.jpg,<class_id>

Image paths are resolved relative to ``image_dir``::

    <image_dir>/<StyleFolder>/<artist_painting>.jpg
"""

from __future__ import annotations

import logging
import pathlib
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(class_file: str | pathlib.Path) -> List[str]:
    """Parse a class-index file and return an ordered list of class names.

    Each line of the file must have the format ``<id> <ClassName>``, e.g.::

        0 Abstract_Expressionism
        1 Action_painting

    Args:
        class_file: Path to the class index text file.

    Returns:
        List of class name strings sorted by their integer index, so that
        ``names[i]`` is the class name for label ``i``.

    Raises:
        FileNotFoundError: If ``class_file`` does not exist.
        ValueError: If a line cannot be parsed as ``<int> <str>``.
    """
    path = pathlib.Path(class_file)
    entries: Dict[int, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Cannot parse line {line!r} in {path}. "
                    "Expected format: '<int> <ClassName>'."
                )
            entries[int(parts[0])] = parts[1]
    return [entries[i] for i in sorted(entries)]


def _parse_csv(csv_path: pathlib.Path) -> Dict[str, int]:
    """Read a WikiArt CSV and return a mapping of relative path → label int.

    Args:
        csv_path: Path to a CSV file with rows ``<rel_path>,<class_id>``.

    Returns:
        Dict mapping relative image path strings to integer class labels.
    """
    mapping: Dict[str, int] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rel_path, label_str = line.rsplit(",", 1)
            mapping[rel_path] = int(label_str)
    return mapping


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WikiArtDataset(Dataset):
    """PyTorch Dataset for multi-attribute WikiArt classification.

    Builds the union of all image paths found across the three CSV files
    (style, genre, artist).  For each image, labels that are not present in
    a given CSV are set to ``fallback_label`` (-1 by default), which the
    :class:`~src.training.losses.MultiTaskLoss` uses as an ignore mask.

    Args:
        image_dir: Root directory containing the WikiArt image folders
            (e.g. ``data/wikiart/images``).
        csv_files: Dict mapping attribute name to CSV path, e.g.::

                {"style": "data/wikiart/style_train.csv",
                 "genre": "data/wikiart/genre_train.csv",
                 "artist": "data/wikiart/artist_train.csv"}

        transform: Optional callable applied to each ``PIL.Image`` before it
            is returned.  Should output a ``torch.Tensor``.
        fallback_label: Integer label used when an image is absent from a
            particular attribute CSV (default -1).

    Example::

        ds = WikiArtDataset(
            image_dir="data/wikiart/images",
            csv_files={
                "style":  "data/wikiart/style_train.csv",
                "genre":  "data/wikiart/genre_train.csv",
                "artist": "data/wikiart/artist_train.csv",
            },
            transform=get_train_transforms(),
        )
        image, labels = ds[0]
        # image : Tensor (3, 224, 224)
        # labels: {"style": int, "genre": int, "artist": int}
    """

    def __init__(
        self,
        image_dir: str | pathlib.Path,
        csv_files: Dict[str, str | pathlib.Path],
        transform: Optional[Callable] = None,
        fallback_label: int = -1,
    ) -> None:
        self.image_dir = pathlib.Path(image_dir)
        self.transform = transform
        self.fallback_label = fallback_label

        # Parse all CSVs: attribute → {rel_path: label}
        self._label_maps: Dict[str, Dict[str, int]] = {
            attr: _parse_csv(pathlib.Path(csv_path))
            for attr, csv_path in csv_files.items()
        }

        # Union of all relative image paths, sorted for reproducibility
        all_paths: set[str] = set()
        for mapping in self._label_maps.values():
            all_paths.update(mapping.keys())
        self._rel_paths: List[str] = sorted(all_paths)

        self._attributes: List[str] = list(self._label_maps.keys())

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of unique images across all CSVs."""
        return len(self._rel_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Return a single (image_tensor, labels_dict) pair.

        If the image file is not found on disk a black RGB image is returned
        instead so training can continue uninterrupted.

        Args:
            index: Integer index into the dataset.

        Returns:
            A 2-tuple ``(image_tensor, labels_dict)`` where:

            - **image_tensor** — shape ``(3, H, W)`` after transform, or
              ``(3, 224, 224)`` black image on file-not-found.
            - **labels_dict** — ``{"style": int, "genre": int, "artist": int}``
              with ``fallback_label`` for missing attributes.
        """
        rel_path = self._rel_paths[index]
        abs_path = self.image_dir / rel_path

        try:
            image = Image.open(abs_path).convert("RGB")
        except FileNotFoundError:
            logger.warning("Image not found, substituting black image: %s", abs_path)
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform is not None:
            image_tensor: torch.Tensor = self.transform(image)
        else:
            from torchvision import transforms as T
            image_tensor = T.ToTensor()(image)

        labels: Dict[str, int] = {
            attr: self._label_maps[attr].get(rel_path, self.fallback_label)
            for attr in self._attributes
        }
        return image_tensor, labels

    # ------------------------------------------------------------------
    # Sampler helpers
    # ------------------------------------------------------------------

    def get_sample_weights(self, attribute: str) -> List[float]:
        """Compute per-sample weights for ``WeightedRandomSampler``.

        Only samples that have a valid (non-fallback) label for ``attribute``
        contribute to the class counts.  Samples with a fallback label
        receive a weight of 0, so they are never selected by the sampler.

        Formula (for valid samples)::

            weight_i = total_valid / (num_classes × class_count[label_i])

        Args:
            attribute: One of the attribute names present in ``csv_files``
                (e.g. ``"style"``).

        Returns:
            List of floats of length ``len(self)``, one weight per sample.

        Raises:
            KeyError: If ``attribute`` is not a key in ``csv_files``.
        """
        if attribute not in self._label_maps:
            raise KeyError(
                f"Attribute {attribute!r} not found. "
                f"Available attributes: {list(self._label_maps.keys())}"
            )

        label_map = self._label_maps[attribute]

        # Collect valid labels (non-fallback)
        valid_labels: List[int] = [
            label_map.get(rel, self.fallback_label)
            for rel in self._rel_paths
        ]
        valid_only = [l for l in valid_labels if l != self.fallback_label]
        total_valid = len(valid_only)

        if total_valid == 0:
            return [0.0] * len(self._rel_paths)

        num_classes = len(set(valid_only))
        class_count: Dict[int, int] = {}
        for lbl in valid_only:
            class_count[lbl] = class_count.get(lbl, 0) + 1

        weights: List[float] = []
        for lbl in valid_labels:
            if lbl == self.fallback_label:
                weights.append(0.0)
            else:
                w = total_valid / (num_classes * class_count[lbl])
                weights.append(w)
        return weights
