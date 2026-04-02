<div align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</div>

<h1 align="center">Breaking Fake :triangular_flag_on_post:</h1>
<p align="center"><b>A Tri-Factor Forensic Engine for Deepfake & Generative AI Image Detection</b></p>
<hr>

## 🔍 Overview
**Breaking Fake** is a state-of-the-art Generative AI detection web application. Unlike standard detectors that rely on a single neural network (which can be easily tricked by compression), Breaking Fake utilizes a **"3-Judge Integration System"** to analyze images from spatial, frequency, and metadata perspectives.

It is built to specifically handle the **"WhatsApp Problem"** (where social media platforms strip metadata upon image upload). Even without digital passports, our spatial and frequency models determine authenticity.

## ⚙️ The 3-Judge Architecture (Result Fusion Engine)
Our deployment runs three independent mathematical processes under the hood and averages their result for an all-encompassing **AI Confidence %**:

1. **The Spatial Anatomy Judge (ViT) `[60% Weight]`**
   * Uses a custom-trained Vision Transformer (`ViT`) to seek out spatial anomalies and generative grid patterns.
   * Outputs a **Grad-CAM XAI Heatmap** to show you exactly *where* the AI flagged the artificial anatomy.

2. **The Frequency Matrix Judge (FFT) `[30% Weight]`**
   * Employs Fast Fourier Transforms (`FFT`) to convert the image into a grayscale frequency map. 
   * Real images have continuous noise signatures; Generative models create mathematical clusters and uniform variances due to upscaling processes.

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

<hr>
<p align="center"><i>Building trust in the era of Generative AI.</i></p>
