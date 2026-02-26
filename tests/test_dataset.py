"""Unit tests for WikiArtDataset and load_class_names.

All tests use the actual CSV and class files that already exist under
``data/wikiart/``, so no mocking of file I/O is required for the CSV /
class-file tests.  Image loading tests use a tiny synthetic dataset.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest
import torch

from src.data.dataset import WikiArtDataset, load_class_names
from src.data.transforms import get_val_transforms

# Paths relative to the project root (tests run from the project root)
DATA_DIR = pathlib.Path("data/wikiart")
IMAGE_DIR = DATA_DIR / "images"

CSV_FILES_TRAIN = {
    "style": str(DATA_DIR / "style_train.csv"),
    "genre": str(DATA_DIR / "genre_train.csv"),
    "artist": str(DATA_DIR / "artist_train.csv"),
}


# ---------------------------------------------------------------------------
# load_class_names
# ---------------------------------------------------------------------------

class TestLoadClassNames:
    def test_style_class_count(self):
        names = load_class_names(DATA_DIR / "style_class.txt")
        assert len(names) == 27

    def test_genre_class_count(self):
        names = load_class_names(DATA_DIR / "genre_class.txt")
        assert len(names) == 10

    def test_artist_class_count(self):
        names = load_class_names(DATA_DIR / "artist_class.txt")
        assert len(names) == 23

    def test_names_are_strings(self):
        names = load_class_names(DATA_DIR / "style_class.txt")
        assert all(isinstance(n, str) for n in names)

    def test_first_style_is_abstract_expressionism(self):
        names = load_class_names(DATA_DIR / "style_class.txt")
        assert names[0] == "Abstract_Expressionism"


# ---------------------------------------------------------------------------
# WikiArtDataset length and label keys
# ---------------------------------------------------------------------------

class TestWikiArtDatasetBasics:
    @pytest.fixture(scope="class")
    def dataset(self):
        return WikiArtDataset(
            image_dir=IMAGE_DIR,
            csv_files=CSV_FILES_TRAIN,
            transform=get_val_transforms(),
        )

    def test_dataset_nonempty(self, dataset):
        assert len(dataset) > 0

    def test_len_matches_union_of_csvs(self, dataset):
        # The union must be at least as large as the largest single CSV
        # Style train has 57,025 rows — union >= 57,025
        assert len(dataset) >= 57_025

    def test_getitem_returns_tuple(self, dataset):
        item = dataset[0]
        assert isinstance(item, tuple) and len(item) == 2

    def test_image_tensor_shape(self, dataset):
        image, _ = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)

    def test_labels_dict_keys(self, dataset):
        _, labels = dataset[0]
        assert set(labels.keys()) == {"style", "genre", "artist"}

    def test_labels_are_ints(self, dataset):
        _, labels = dataset[0]
        for v in labels.values():
            assert isinstance(v, int)


# ---------------------------------------------------------------------------
# Fallback label for missing attributes
# ---------------------------------------------------------------------------

class TestFallbackLabel:
    def _make_minimal_dataset(self, tmp_path: pathlib.Path) -> WikiArtDataset:
        """Create a dataset where only the style CSV has an entry."""
        # Write a style CSV with one image
        style_csv = tmp_path / "style.csv"
        style_csv.write_text("FakeStyle/img.jpg,0\n", encoding="utf-8")

        # Genre CSV is empty (no matching entry for img.jpg)
        genre_csv = tmp_path / "genre.csv"
        genre_csv.write_text("", encoding="utf-8")

        # Artist CSV has a different image
        artist_csv = tmp_path / "artist.csv"
        artist_csv.write_text("FakeStyle/other.jpg,1\n", encoding="utf-8")

        return WikiArtDataset(
            image_dir=tmp_path,
            csv_files={
                "style": str(style_csv),
                "genre": str(genre_csv),
                "artist": str(artist_csv),
            },
            fallback_label=-1,
        )

    def test_missing_genre_returns_fallback(self, tmp_path):
        ds = self._make_minimal_dataset(tmp_path)
        # Find the index for FakeStyle/img.jpg
        idx = ds._rel_paths.index("FakeStyle/img.jpg")
        _, labels = ds[idx]
        assert labels["genre"] == -1

    def test_present_label_is_not_fallback(self, tmp_path):
        ds = self._make_minimal_dataset(tmp_path)
        idx = ds._rel_paths.index("FakeStyle/img.jpg")
        _, labels = ds[idx]
        assert labels["style"] == 0

    def test_missing_artist_returns_fallback(self, tmp_path):
        ds = self._make_minimal_dataset(tmp_path)
        idx = ds._rel_paths.index("FakeStyle/img.jpg")
        _, labels = ds[idx]
        assert labels["artist"] == -1


# ---------------------------------------------------------------------------
# get_sample_weights
# ---------------------------------------------------------------------------

class TestGetSampleWeights:
    @pytest.fixture(scope="class")
    def dataset(self):
        return WikiArtDataset(
            image_dir=IMAGE_DIR,
            csv_files=CSV_FILES_TRAIN,
        )

    def test_weights_length_matches_dataset(self, dataset):
        weights = dataset.get_sample_weights("style")
        assert len(weights) == len(dataset)

    def test_weights_are_non_negative(self, dataset):
        weights = dataset.get_sample_weights("style")
        assert all(w >= 0.0 for w in weights)

    def test_invalid_attribute_raises(self, dataset):
        with pytest.raises(KeyError):
            dataset.get_sample_weights("nonexistent")
