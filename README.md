# DeepArtNet 🎨

> **Hierarchical Art Attribute Recognition Using CNN-RNN Architectures on the WikiArt/ArtGAN Dataset**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-WikiArt%2FArtGAN-purple.svg)](https://github.com/cs-chan/ArtGAN)

---

## 📖 Overview

**DeepArtNet** is a deep learning system for **multi-attribute fine art classification**, recognizing three core attributes simultaneously from painting images:

| Attribute | Classes | Train Samples | Val Samples |
|-----------|---------|---------------|-------------|
| 🖌️ **Style** | 27 | 57,025 | 24,421 |
| 🏛️ **Genre** | 10 | 45,503 | 19,492 |
| 👤 **Artist** | 23 | 13,346 | 5,706 |

The model uses a **hybrid CNN-RNN architecture**: an EfficientNet-B4 backbone extracts spatial features, which are then sequenced and processed by a Bidirectional LSTM with additive attention — capturing both local brushstroke detail and long-range compositional structure.

---

## 🗂️ Table of Contents

- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [References](#-references)

---

## 📊 Dataset

This project uses the **WikiArt/ArtGAN** dataset, provided as pre-split CSV files.

### Included CSV Files (`data/wikiart/`)

| File | Rows | Description |
|------|------|-------------|
| `style_train.csv` | 57,025 | Training set for style classification |
| `style_val.csv` | 24,421 | Validation set for style classification |
| `genre_train.csv` | 45,503 | Training set for genre classification |
| `genre_val.csv` | 19,492 | Validation set for genre classification |
| `artist_train.csv` | 13,346 | Training set for artist classification |
| `artist_val.csv` | 5,706 | Validation set for artist classification |
| `style_class.txt` | 27 | Style class index → name mapping |
| `genre_class.txt` | 10 | Genre class index → name mapping |
| `artist_class.txt` | 23 | Artist class index → name mapping |

### CSV Format

Each CSV row: `<StyleFolder>/<artist>_<painting>.jpg,<class_id>`

```
Impressionism/edgar-degas_landscape-on-the-orne.jpg,12
Realism/camille-corot_mantes-cathedral.jpg,21
Abstract_Expressionism/gene-davis_untitled-1979(3).jpg,0
```

### Style Classes (27)
`Abstract_Expressionism`, `Action_painting`, `Analytical_Cubism`, `Art_Nouveau`, `Baroque`, `Color_Field_Painting`, `Contemporary_Realism`, `Cubism`, `Early_Renaissance`, `Expressionism`, `Fauvism`, `High_Renaissance`, `Impressionism`, `Mannerism_Late_Renaissance`, `Minimalism`, `Naive_Art_Primitivism`, `New_Realism`, `Northern_Renaissance`, `Pointillism`, `Pop_Art`, `Post_Impressionism`, `Realism`, `Rococo`, `Romanticism`, `Symbolism`, `Synthetic_Cubism`, `Ukiyo_e`

### Genre Classes (10)
`abstract_painting`, `cityscape`, `genre_painting`, `illustration`, `landscape`, `nude_painting`, `portrait`, `religious_painting`, `sketch_and_study`, `still_life`

### Artist Classes (23)
`Albrecht_Durer`, `Boris_Kustodiev`, `Camille_Pissarro`, `Childe_Hassam`, `Claude_Monet`, `Edgar_Degas`, `Eugene_Boudin`, `Gustave_Dore`, `Ilya_Repin`, `Ivan_Aivazovsky`, `Ivan_Shishkin`, `John_Singer_Sargent`, `Marc_Chagall`, `Martiros_Saryan`, `Nicholas_Roerich`, `Pablo_Picasso`, `Paul_Cezanne`, `Pierre_Auguste_Renoir`, `Pyotr_Konchalovsky`, `Raphael_Kirchner`, `Rembrandt`, `Salvador_Dali`, `Vincent_van_Gogh`

### Class Imbalance Note

Style has severe imbalance: Impressionism (9,142) vs Synthetic_Cubism (152) — a 60:1 ratio. The trainer uses **Focal Loss** and **WeightedRandomSampler** to address this.

### Image Download

The CSV files are included in this repo. Images must be downloaded separately from WikiArt:

```bash
python scripts/download_images.py --output data/wikiart/images/
```

Images should be placed at: `data/wikiart/images/<StyleFolder>/<artist>_<painting>.jpg`

---

## 🧠 Architecture

```
Input Image (B, 3, 224, 224)
         │
         ▼
┌──────────────────────────┐
│  EfficientNet-B4 (CNN)   │  pretrained ImageNet
│  + Conv1×1 Projection    │  Output: (B, 512, 7, 7)
└──────────────────────────┘
         │
         ▼  reshape
┌──────────────────────────┐
│  Spatial Sequencer       │  (B, 49, 512)
│  7×7 grid → 49 tokens    │  each cell = one timestep
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Bidirectional LSTM ×2   │  hidden=256 per direction
│                          │  Output: (B, 49, 512)
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Additive Attention      │  Bahdanau-style
│                          │  context: (B, 512)
│                          │  weights: (B, 49) → visualizable
└──────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│           Multi-Task Classification Heads    │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐  │
│  │  Style  │   │  Genre  │   │  Artist  │  │
│  │ 27 cls  │   │ 10 cls  │   │  23 cls  │  │
│  └─────────┘   └─────────┘   └──────────┘  │
└─────────────────────────────────────────────┘
```

Each classification head: `Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→N)`

---

## 📁 Project Structure

```
DeepArtNet/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── base_config.yaml          # Model + data hyperparameters
│   └── train_config.yaml         # Phase-wise training settings
│
├── data/
│   └── wikiart/
│       ├── images/               # ← Place downloaded WikiArt images here
│       │   └── <StyleFolder>/
│       │       └── <artist>_<painting>.jpg
│       ├── style_train.csv       # ✅ Included — 57,025 rows
│       ├── style_val.csv         # ✅ Included — 24,421 rows
│       ├── genre_train.csv       # ✅ Included — 45,503 rows
│       ├── genre_val.csv         # ✅ Included — 19,492 rows
│       ├── artist_train.csv      # ✅ Included — 13,346 rows
│       ├── artist_val.csv        # ✅ Included —  5,706 rows
│       ├── style_class.txt       # ✅ Included — 27 style names
│       ├── genre_class.txt       # ✅ Included — 10 genre names
│       └── artist_class.txt      # ✅ Included — 23 artist names
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_backbone.py       # EfficientNet-B4 + projection head
│   │   ├── bilstm_encoder.py     # 2-layer Bidirectional LSTM
│   │   ├── attention.py          # Bahdanau additive attention
│   │   ├── classification_heads.py  # Multi-task MLP heads
│   │   └── deepartnet.py         # Full model assembly + predict()
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # WikiArtDataset (reads CSVs directly)
│   │   ├── transforms.py         # Train / val augmentation pipelines
│   │   └── dataloader.py         # build_dataloaders(), WeightedSampler
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # 3-phase Trainer class
│   │   ├── losses.py             # FocalLoss + MultiTaskLoss (Kendall 2018)
│   │   └── scheduler.py          # Cosine annealing LR scheduler
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Top-1/5 accuracy, confusion matrix
│   │   └── evaluator.py          # Evaluation loop with per-class stats
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py      # Attention heatmap overlays
│       ├── checkpoint.py         # Save / load checkpoints
│       └── logging_utils.py      # TensorBoard + console logging
│
├── scripts/
│   ├── train.py                  # Main CLI training script
│   ├── evaluate.py               # Evaluation script
│   ├── inference.py              # Single-image prediction
│   └── download_images.py        # WikiArt image downloader helper
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_architecture.ipynb
│   ├── 03_training_curves.ipynb
│   └── 04_attention_visualization.ipynb
│
├── tests/
│   ├── test_dataset.py           # WikiArtDataset unit tests
│   ├── test_model.py             # Forward pass shape tests
│   └── test_losses.py            # FocalLoss / MultiTaskLoss tests
│
└── outputs/
    ├── checkpoints/              # Saved .pth model weights
    ├── logs/                     # TensorBoard event files
    └── visualizations/           # Attention map images
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/DeepArtNet.git
cd DeepArtNet

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Verify
python -c "from src.models import DeepArtNet; print('Ready')"
```

---

## 🚀 Usage

### Train (all 3 phases)
```bash
python scripts/train.py --config configs/train_config.yaml
```

### Train a single phase
```bash
python scripts/train.py --phase 1 --epochs 20 --lr 1e-3
python scripts/train.py --phase 2 --epochs 30 --lr 5e-4 --resume outputs/checkpoints/phase1_best.pth
python scripts/train.py --phase 3 --epochs 20 --lr 1e-5 --resume outputs/checkpoints/phase2_best.pth
```

### Evaluate
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

### Single image inference
```bash
python scripts/inference.py --checkpoint outputs/checkpoints/best_model.pth \
    --image path/to/painting.jpg --visualize_attention
```

### Python API
```python
from src.models import DeepArtNet
from PIL import Image

model = DeepArtNet.load_from_checkpoint("outputs/checkpoints/best_model.pth")
image = Image.open("painting.jpg")
results = model.predict(image)

# {'style': {'label': 'Impressionism', 'confidence': 0.87},
#  'genre': {'label': 'landscape',     'confidence': 0.91},
#  'artist':{'label': 'Claude_Monet',  'confidence': 0.73}}
```

---

## 🏋️ Training Strategy

Three progressive phases, each building on the previous:

| Phase | Frozen | LR | Epochs | Batch | Purpose |
|-------|--------|----|--------|-------|---------|
| **1** | LSTM + Attention | 1e-3 | 20 | 64 | Warm up CNN features |
| **2** | Backbone blocks 0–2 | 5e-4 | 30 | 32 | Joint CNN-RNN learning |
| **3** | Nothing | 1e-5 | 20 | 32 | Full end-to-end fine-tuning |

### Multi-Task Loss

```
L_total = Σᵢ [ 1/(2σᵢ²) · Lᵢ + log(σᵢ) ]
```

Each `Lᵢ` is **Focal Loss** (γ=2) to handle class imbalance. The σᵢ per task are **learned parameters** that automatically balance task contributions (Kendall et al., 2018).

---

## 📈 Expected Results

| Attribute | Top-1 Acc | Top-5 Acc |
|-----------|-----------|-----------|
| Style (27 cls) | ~76% | ~94% |
| Genre (10 cls) | ~83% | ~97% |
| Artist (23 cls) | ~72% | ~93% |

---

## 📚 References

1. Tan et al. (2017). *ArtGAN: Artwork Synthesis with Conditional Categorical GANs.* [arXiv:1702.03410](https://arxiv.org/abs/1702.03410)
2. Saleh & Elgammal (2015). *Large-scale Classification of Fine-Art Paintings.* [arXiv:1505.00855](https://arxiv.org/abs/1505.00855)
3. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
4. Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
5. Kendall et al. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses.* [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
6. Lin et al. (2017). *Focal Loss for Dense Object Detection.* [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center"><strong>DeepArtNet</strong> — Teaching machines to see art the way humans do 🎨</p>