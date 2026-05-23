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

3. **The Digital Passport Judge (Metadata) `[10% Weight]`**
   * Explores the raw `EXIF` datablocks.
   * Checks for C2PA content credentials (Adobe verification keys), or verifies legitimate hardware lens fingerprints.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Breaking-Fake.git
   cd Breaking-Fake
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit UI:
   ```bash
   streamlit run app.py
   ```
   *(Note: The ViT model `.pth` file is too large for GitHub and will auto-download from Google Drive upon first initialization.)*

## 🌐 Deploying to Streamlit Cloud

To push this exact branch to **[Streamlit Community Cloud](https://streamlit.io/)**:
1. Upload this codebase to a public GitHub repository. Ensure `.gitignore` is intact so the large `.pth` model file is excluded.
2. Visit Streamlit, link your GitHub, and select `app.py` as the entrypoint. 
3. *That's it!* The `gdown` dependency in `requirements.txt` will automatically pull the model into Streamlit's virtual machine.

## 💡 The Data Workflow (2-Person Split)
Designed collaboratively by splitting the stack into:
- **Member A (Brain):** ViT Model Training, Grad-CAM, Heatmaps.
- **Member B (Architecture):** FFT Matrices, Streamlit Web Framework, Metadata Scripts, and UX Result Fusion.

<hr>
<p align="center"><i>Building trust in the era of Generative AI.</i></p>
