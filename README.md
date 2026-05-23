<div align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</div>

<h1 align="center">Breaking Fake :triangular_flag_on_post:</h1>
<p align="center"><b>A Tri-Factor Forensic Engine for Deepfake & Generative AI Image Detection</b></p>
<hr>

## 🔍 Overview

**Breaking Fake** is a state-of-the-art Generative AI detection web application. Unlike standard detectors relying on a single neural network (easily fooled by compression), Breaking Fake uses a **"3-Judge Integration System"** analyzing images from spatial, frequency, and metadata perspectives.

Built to handle the **"WhatsApp Problem"** (metadata stripping on social media). Even without EXIF data, our spatial and frequency models determine authenticity.

## ⚙️ The 3-Judge Architecture

1. **Spatial Anatomy Judge (ViT) [60% weight]**
   - Vision Transformer detects spatial anomalies and generative grid patterns
   - Grad-CAM XAI heatmap shows exact locations of suspicious artifacts

2. **Frequency Matrix Judge (FFT) [30% weight]**
   - Fast Fourier Transform analyzes frequency domain
   - Real images have continuous noise; AI images show grid clusters from upscaling

3. **Digital Passport Judge (Metadata) [10% weight]**
   - Checks EXIF data and C2PA content credentials
   - Verifies hardware lens fingerprints

## 📁 Project Structure (Production-Ready)

```
backend/                    # Inference API & Web Server
├── app/
│   ├── main.py            # Streamlit UI (moved from root app.py)
│   └── __init__.py
├── requirements.txt       # Web server dependencies
└── __init__.py

model/                      # ML Training & Artifacts
├── src/
│   ├── train.py           # Training loop
│   ├── data.py            # Dataset class
│   ├── prepare_data.py    # Data setup helper
│   └── README.md          # Training guide
├── artifacts/             # Model weights (.pth files)
├── requirements.txt       # ML dependencies (torch, timm, etc.)
└── notebooks/            # Experimental analysis (optional)

frontend/                   # React/Vue frontend (optional)
├── README.md

infra/                      # Docker & deployment
├── backend.Dockerfile

scripts/                    # Helper scripts
├── run-backend.sh
└── run-backend.ps1

configs/                    # Configuration files
└── config.prod.yaml

docs/
├── MIGRATION.md           # Migration checklist

# Root files
app.py                      # Shim: imports backend/app/main.py
requirements.txt           # Convenience (all deps)
packages.txt              # System dependencies
.gitignore
README.md
```

## 🚀 Quick Start

### Run Inference (Using Pre-Trained Model)

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run the Streamlit UI
streamlit run app.py
```

Visit `http://localhost:8501` and upload an image to analyze.

**Note**: The `.pth` model auto-downloads from Google Drive on first run (configured in `backend/app/main.py`).

### Train Your Own Model

See [model/src/README.md](model/src/README.md) for detailed training instructions.

**Quick version:**

```bash
# 1. Prepare data
python -m model.src.prepare_data --data-dir data/raw

# 2. Populate folders
# - Place AI-generated images in: data/raw/ai_images/
# - Place real photos in: data/raw/real_images/

# 3. Install ML dependencies
pip install -r model/requirements.txt

# 4. Train the model
python -m model.src.train --data-dir data/raw --epochs 20 --device cuda
```

The best model is saved to `model/artifacts/breaking_fake_vit.pth` and automatically loaded by the app.

## 🐳 Docker Deployment

### Build
```bash
docker build -t breakingfake-backend:latest -f infra/backend.Dockerfile .
```

### Run
```bash
docker run -p 8501:8501 breakingfake-backend:latest
```

Visit `http://localhost:8501`.

## 📊 Training Dataset

For best results, use balanced datasets:

| Category | Recommended | Source |
|----------|-----------|--------|
| AI Images | 10K+ | CIFAKE, DiffusionDB, DALL-E, Midjourney outputs |
| Real Photos | 10K+ | Photographer cameras, Unsplash, Flickr |

For quick testing, start with 100-200 images per category.

## 🏗️ Architecture Details

**Model**: Vision Transformer (ViT Base)
- Backbone: `timm.create_model('vit_base_patch16_224', pretrained=True)`
- Head: 2-class classifier (AI vs. Real)
- Input: 224×224 RGB images
- Optimization: AdamW + Cosine Annealing

**Data Pipeline**: 80/20 train/val split (seed=42 for reproducibility)

**Inference**: CPU-friendly (uses `torch.device('cpu')` for Streamlit Cloud compatibility)

## 💡 Collaborative Design

Designed for team workflows:
- **Backend Engineer**: Manages `backend/`, Docker, deployment
- **ML Engineer**: Owns `model/src/`, training, data curation
- **Frontend Engineer**: Builds SPA in `frontend/` (React/Vue)

## 🐛 Troubleshooting

### Model not loading
Ensure `model/artifacts/breaking_fake_vit.pth` exists or that your Google Drive ID is correct in `backend/app/main.py`.

### Out of memory during training
```bash
python -m model.src.train --batch-size 16 --device cpu
```

## 🔗 References

- [ViT: Vision Transformers](https://arxiv.org/abs/2010.11929)
- [CIFAKE Dataset](https://github.com/peterwang512/CIFAKE)
- [timm Library](https://timm.fast.ai/)
- [Streamlit Docs](https://docs.streamlit.io/)

<hr>
<p align="center"><i>Building trust in the era of Generative AI.</i></p>
