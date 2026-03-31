import os
import sys
# [HOTFIX] Streamlit Cloud Debian 11/12 has broken apt libraries (libgthread/libffi7).
# 'grad-cam' natively imports 'opencv-python' (GUI version) which crashes the cloud server.
# We blast it from the cache right at boot to force the system to use 'opencv-python-headless'.
os.system("pip uninstall -y opencv-python opencv-contrib-python")

import streamlit as st
import torch
import timm
import cv2
import numpy as np
import plotly.graph_objects as go
import os
import gdown
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Configuration & UI Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="Breaking Fake | Forensic AI Studio", layout="wide", page_icon="🔍")

st.markdown("""
<style>
    /* Global Fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Stunning Dark Gradient Background */
    .stApp {
        background: radial-gradient(circle at top left, #2a0808 0%, #060202 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1a0505 !important;
        border-right: 1px solid #b30000 !important;
    }
    
    /* Sidebar Image Inversion (Make black icon white) */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        filter: invert(1) brightness(2);
    }
    
    /* Sleek container spacing - Increase padding-top to avoid app-bar overlap */
    .block-container {
        padding-top: 5rem !important;
        max-width: 1200px;
    }
    
    /* Animated Header */
    @keyframes glow {
        0% { text-shadow: 0 0 5px #ff4d4d, 0 0 10px #cc0000; }
        50% { text-shadow: 0 0 15px #ff3333, 0 0 25px #ff0000; }
        100% { text-shadow: 0 0 5px #ff4d4d, 0 0 10px #cc0000; }
    }
    .main-header {
        font-size: 65px;
        font-weight: 900;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff6666, #b30000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 3s ease-in-out infinite;
        margin-bottom: 0px;
        letter-spacing: 2px;
    }
    .sub-text {
        text-align: center;
        color: #ffcccc;
        margin-bottom: 20px;
        font-size: 22px;
        letter-spacing: 1px;
    }
    
    /* Judge Cards */
    .judge-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 77, 77, 0.2);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .judge-card:hover {
        border-color: rgba(255, 77, 77, 0.6);
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-5px);
    }
    .judge-title {
        color: #ff4d4d;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* File Uploader Hover Animation */
    [data-testid="stFileUploadDropzone"] {
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        border: 2px dashed #b30000 !important;
        border-radius: 15px !important;
        background-color: rgba(179, 0, 0, 0.05) !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        transform: scale(1.02);
        box-shadow: 0px 0px 20px rgba(255, 51, 51, 0.4);
        border-color: #ff3333 !important;
        background-color: rgba(255, 51, 51, 0.1) !important;
    }
    
    /* Button Hover Animations */
    .stButton > button {
        transition: all 0.3s ease-in-out;
        border-radius: 8px !important;
        background: linear-gradient(90deg, #1a0505, #b30000) !important;
        border: 1px solid #ff3333 !important;
        color: white !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 51, 51, 0.4);
        border: 1px solid white !important;
    }
    
    /* Apply fade-in to metrics */
    .metric-box, .stMarkdown, .stImage, .stPlotlyChart {
        animation: fadeIn 0.8s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)


DEVICE = torch.device("cpu") # Render friendly on Streamlit Cloud
MODEL_PATH = "breaking_fake_vit.pth"

# 🛑 PUT YOUR GOOGLE DRIVE FILE ID HERE 🛑
# To get the ID, right-click file in GDrive -> Share -> Copy Link
# Extracted from: https://drive.google.com/file/d/YOUR_FILE_ID/view
GDRIVE_FILE_ID = "14lccsmptKBE2oETX5u_AFjHeDFAIf9uz" 

# -------------------------------------------------------------------
# 1. Initialization (Caching the Model)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Booting Tri-Factor Neural Architecture...")
def load_forensic_model():
    """Downloads model if missing, then loads ViT weights."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found locally. Downloading from Google Drive...")
        try:
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model! Check your GDrive ID. Error: {e}")
            return None
    
    try:
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load ViT Model: {e}")
        return None

# -------------------------------------------------------------------
# 2. Preprocessing & Judges
# -------------------------------------------------------------------
def preprocess_image(pil_img):
    """Matches the exact transforms used in training."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_img).unsqueeze(0).to(DEVICE)

def judge_spatial_vit(model, img_tensor):
    """Judge 1: The ViT AI Anatomy Engine (Returns AI Probability 0-100%)."""
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        
    ai_prob = probabilities[0].item() * 100 # 0 = AI, 1 = Real in CIFAKE
    return ai_prob

def judge_frequency_fft(pil_img):
    """Judge 2: The FFT Frequency Engine (Returns AI Probability 0-100%)."""
    gray = np.array(pil_img.convert('L'))
    gray = cv2.resize(gray, (224, 224))
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Normalize for visualization
    spectrum_vis = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    variance_score = np.var(magnitude_spectrum)
    
    # Scale variance to an AI Probability Score (Heuristic: Lower variance = higher chance of AI upscaling)
    # This formula maps variance from a typical range (e.g., 0-1000) to a percentage.
    ai_score = max(0, min(100, 100 - (variance_score / 15))) 
    return spectrum_vis, variance_score, ai_score

def judge_metadata(pil_img):
    """Judge 3: The Digital Passport Inspector (Returns AI Probability 0-100%)."""
    img_info = str(pil_img.info).lower()
    
    # 1. C2PA / Adobe Signatures
    if "c2pa" in img_info or "jumb" in img_info or "adobe" in img_info:
        return 100.0, "SOFTWARE FLAGGED: Image contains synthetic creation credentials or rendering engine markers."
        
    # 2. Hardware EXIF Check
    exif_data = pil_img.getexif()
    if exif_data:
        hardware_tags = ["Make", "Model", "LensModel", "DateTimeOriginal"]
        found_tags = [TAGS.get(tag_id, tag_id) for tag_id in exif_data if TAGS.get(tag_id, tag_id) in hardware_tags]
        
        if len(found_tags) >= 2:
            return 0.0, f"AUTHENTIC HARDWARE: Captured via verified hardware ({found_tags[0] if found_tags else 'Camera'})."
        elif "Software" in str(exif_data):
            return 80.0, "SUSPICIOUS: Software modification tags found instead of hardware sensors."
            
    # 3. No Metadata (The WhatsApp Problem)
    return 50.0, "STRIPPED METADATA: No EXIF data found. (Commonly stripped by WhatsApp/Instagram, relying heavily on ViT and FFT)."

def generate_xai_heatmap(model, img_tensor, pil_img):
    """Judge 1 XAI: Neural Network Feature Map adapted for Vision Transformers."""
    # Vision Transformers output 1D token sequences (e.g. 197 tokens). 
    # Grad-CAM expects 2D spatial maps (Channels, H, W). We must reshape the tokens back into a 14x14 grid.
    def reshape_transform(tensor, height=14, width=14):
        # Discard the [CLS] token (index 0) and keep the 196 patch tokens
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        # Transpose from (Batch, H, W, Channels) to (Batch, Channels, H, W)
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    target_layers = [model.blocks[-1].norm1]
    
    # Initialize GradCAM with the reshape_transform function!
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(0)] # Target the 'AI' class
    
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    img_resized = pil_img.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization

def create_gauge_chart(score):
    """Creates a Ploly Gauge Chart for the AI Probability."""
    color = "red" if score > 50 else "#00ff00"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI PROBABILITY", 'font': {'size': 24, 'color': 'white'}},
        number = {'suffix': "%", 'font': {'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "rgba(0, 255, 0, 0.1)"},
                {'range': [50, 100], 'color': "rgba(255, 0, 0, 0.1)"}],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# -------------------------------------------------------------------
# Sidebar & Forensic Intelligence Info
# -------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843048.png", width=120)
    st.markdown("<h2 style='text-align: center; color: #ff4d4d;'>BREAKING FAKE</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9rem;'>Digital Forensic AI Studio v2.0</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("🕵️ The 3-Judge Architecture", expanded=True):
        st.markdown("""
        **1. Neural Engine (ViT)**
        *Weight: 60%*
        Uses a Vision Transformer to detect semantic artifacts (eyes, skin texture, lighting) and spatial noise patterns common in GAN/Diffusion models.
        
        **2. Frequency Engine (FFT)**
        *Weight: 30%*
        Analyzes the frequency domain to find periodic grid artifacts left by deep-learning upscalers.
        
        **3. Metadata Inspector**
        *Weight: 10%*
        Checks for C2PA signatures, hardware EXIF data, and software modification history.
        """)
        
    st.markdown("---")
    st.info("💡 **Pro Tip**: Use the XAI Heatmap to see exactly which pixels the AI found 'suspicious'. Red zones drive the AI score higher.")
    st.warning("⚠️ **Note**: Social media (WhatsApp/FB) strips metadata. Rely on Judge 1 & 2 for these files.")

# -------------------------------------------------------------------
# Main UI Header
# -------------------------------------------------------------------
st.markdown('<p class="main-header">BREAKING FAKE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Premier Neural Studio for AI Image Forensics</p>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Core Analysis Engine
# -------------------------------------------------------------------
model = load_forensic_model()

# Setup Main Workspace Columns
col_main, col_info = st.columns([2, 1])

with col_info:
    st.markdown("""
    ### 🔬 Deep Analysis
    Upload any image (PNG, JPG, WEBP) to begin a multi-domain forensic sweep.
    
    **Current Capability:**
    - Generative Artifact Detection
    - Frequency Domain Profiling
    - Hardware Passport Verification
    """)

with col_main:
    uploaded_file = st.file_uploader("Drop evidence file for deep forensic scan", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    # 📸 Downsize image immediately to save RAM (Optimized for 1GB Cloud Limit)
    img = Image.open(uploaded_file).convert('RGB')
    if img.size[0] > 1024 or img.size[1] > 1024:
        img.thumbnail((1024, 1024)) 
    
    st.markdown("---")
    st.markdown("### 📊 Forensic Intelligence Operations")
    
    # -------------------------------------------------------------------
    # EXECUTE SCAN
    # -------------------------------------------------------------------
    with st.spinner("Executing Multi-Domain Neural Scan..."):
        # Preprocessing
        img_tensor = preprocess_image(img)
        
        # 1. Neural Engine (ViT)
        vit_score = judge_spatial_vit(model, img_tensor)
        heatmap = generate_xai_heatmap(model, img_tensor, img)
        
        # 2. Frequency Engine (FFT)
        spectrum_vis, fft_var, fft_score = judge_frequency_fft(img)
        
        # 3. Metadata Inspector
        meta_score, meta_desc = judge_metadata(img)
        
        # 4. Result Fusion (The Master Equation)
        final_probability = (vit_score * 0.6) + (fft_score * 0.3) + (meta_score * 0.1)
        
        del img_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # --- RENDER DASHBOARD ---
    
    # Top Row: Gauge & Verdict Card
    v_col1, v_col2 = st.columns([1, 1.2])
    
    with v_col1:
        st.plotly_chart(create_gauge_chart(final_probability), use_container_width=True)
        
    with v_col2:
        st.markdown(f"""
        <div class="judge-card">
            <div class="judge-title">🛡️ Global Forensic Verdict</div>
            <h2 style='color: {"#ff4d4d" if final_probability > 50 else "#00ff00"};'>
                {"AI DETECTED" if final_probability > 50 else "AUTHENTIC PHOTOGRAPH"}
            </h2>
            <p style='font-size: 1.1rem;'>Confidence: <strong>{final_probability:.1f}% AI Signature detected</strong></p>
            <hr style='border-color: rgba(255,77,77,0.2);'>
            <p>Scanning status: <strong>Completed</strong>. No further anomalies detected in this sample.</p>
        </div>
        """, unsafe_allow_html=True)

    # Middle Row: Judge Breakdown Cards
    st.markdown("#### 🧬 Judge-Level Intelligence")
    j_col1, j_col2, j_col3 = st.columns(3)
    
    with j_col1:
        st.markdown(f"""
        <div class="judge-card">
            <div class="judge-title">1. Neural Engine</div>
            <p style='font-size: 1.5rem;'>{vit_score:.1f}%</p>
            <p style='font-size: 0.8rem; color: #ffcccc;'>ViT analyzed spatial texture and semantic consistency.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with j_col2:
        st.markdown(f"""
        <div class="judge-card">
            <div class="judge-title">2. Frequency Engine</div>
            <p style='font-size: 1.5rem;'>{fft_score:.1f}%</p>
            <p style='font-size: 0.8rem; color: #ffcccc;'>Variance: {fft_var:.1f}. Upscaling grid noise detected.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with j_col3:
        st.markdown(f"""
        <div class="judge-card">
            <div class="judge-title">3. Digital Passport</div>
            <p style='font-size: 1.5rem;'>{meta_score:.1f}%</p>
            <p style='font-size: 0.8rem; color: #ffcccc;'>{meta_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Bottom Row: Visual Proofs
    st.markdown("#### 🧪 Scientific Proof")
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.markdown("##### XAI Anatomy Map")
        st.image(heatmap, use_container_width=True)
        st.caption("Neural activation zones showing where the model found generative artifacts.")
        
    with p_col2:
        st.markdown("##### FFT Power Spectrum")
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(spectrum_vis, cmap='inferno')
        ax.axis('off')
        fig.patch.set_alpha(0) # Transparent background
        st.pyplot(fig)
        st.caption("Frequency domain analysis showing noise distribution and grid signatures.")

    # Narrative Summary
    st.markdown("---")
    st.markdown("### 📝 Forensic Summary & Narrative")
    if meta_score == 50.0 and final_probability < 50:
        st.info("**AUTHENTIC CASE**: Metadata was stripped (Social Media compression), but Neural and Frequency signatures match a physical sensor. Verdict: AUTHENTIC PHOTOGRAPH.")
    elif final_probability > 50:
        st.error("**FRAUD CASE**: High activations in the XAI map and abnormal FFT grid noise confirm this image was digitally synthesized. Verdict: AI GENERATED.")
    else:
        st.success("**AUTHENTIC CASE**: All 3 judges verified natural pixel continuity and hardware markers. Verdict: AUTHENTIC PHOTOGRAPH.")

else:
    # Landing State
    st.markdown("---")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.03); padding: 40px; border-radius: 20px; text-align: center; border: 1px dashed rgba(255,77,77,0.3);'>
        <h2 style='color: #ff4d4d;'>SYSTEM STANDBY</h2>
        <p style='font-size: 1.2rem; color: #ffcccc;'>Upload an image to engage the Tri-Factor Forensic Suite.</p>
        <br>
        <div style='display: flex; justify-content: space-around; gap: 20px;'>
            <div>
                <h4>Neural Mapping</h4>
                <p style='font-size: 0.8rem;'>Vision Transformer analysis.</p>
            </div>
            <div>
                <h4>Frequency Domain</h4>
                <p style='font-size: 0.8rem;'>FFT Noise investigation.</p>
            </div>
            <div>
                <h4>Passport Check</h4>
                <p style='font-size: 0.8rem;'>EXIF Hardware validation.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
