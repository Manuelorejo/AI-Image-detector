# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:20:46 2025

@author: Oreoluwa
"""

# app.py
# Hybrid AI Image Detection System (Forensic + Sightengine)

import streamlit as st
import requests
from PIL import Image, ExifTags
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG — API KEYS
# =========================
API_USER = "169723322"
API_SECRET = "jBz4EHra243nDi9w449kqfqr2P9GrAdF"

st.set_page_config(page_title="AI Image Detector (Hybrid + Forensics)", layout="centered")
st.title("🖼️ Hybrid AI Image Detection System")
st.write("Upload an image to check if it's AI-generated or real using local forensic analysis and the Sightengine API.")

# =========================
# Metadata extraction
# =========================
def extract_metadata(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image.getexif()
        metadata = {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            metadata[tag] = value
        return metadata
    except Exception:
        return {}

# =========================
# Forensic feature calculators
# =========================
def frequency_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    mag_log = np.log1p(magnitude)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = r.max()

    low_mask = r <= (0.25 * max_r)
    mid_mask = (r > (0.25 * max_r)) & (r <= (0.6 * max_r))
    high_mask = r > (0.6 * max_r)

    total = mag_log.sum() + 1e-8
    high_energy = mag_log[high_mask].sum() / total
    mid_energy = mag_log[mid_mask].sum() / total
    low_energy = mag_log[low_mask].sum() / total

    frac_high = high_energy
    balance = (mid_energy + high_energy) / (low_energy + 1e-8)
    raw = frac_high * 0.7 + (balance / (balance + 1)) * 0.3
    score = float(np.clip(raw, 0.0, 1.0))
    return score, mag_log

def noise_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    med = cv2.medianBlur((gray * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    residual = gray - med
    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
    lap = np.abs(lap)

    resid_std = np.std(residual)
    lap_std = np.std(lap) / 255.0

    hist, _ = np.histogram(residual.ravel(), bins=64, range=(-0.2, 0.2), density=True)
    hist = hist + 1e-12
    entropy = -np.sum(hist * np.log(hist))

    resid_norm = np.tanh(resid_std * 10)
    lap_norm = np.tanh(lap_std * 5)
    ent_norm = np.tanh(entropy / 3.0)
    raw = 0.45 * resid_norm + 0.35 * lap_norm + 0.20 * ent_norm
    score = float(np.clip(raw, 0.0, 1.0))
    return score, residual

def compression_score(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    _, enc = cv2.imencode('.jpg', image, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dec_gray = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY).astype(np.float32)
    error = np.abs(orig - dec_gray) / 255.0
    h, w = error.shape
    bh, bw = 8, 8
    variances = []
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            block = error[y:y+bh, x:x+bw]
            variances.append(np.var(block))
    variances = np.array(variances)
    if variances.size == 0:
        return 0.5, error
    spread = np.std(variances)
    raw = np.tanh(spread * 50)
    score = float(np.clip(raw, 0.0, 1.0))
    return score, error

def color_corr_score(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:,:,0].ravel(), rgb[:,:,1].ravel(), rgb[:,:,2].ravel()
    def safe_corr(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return np.corrcoef(a, b)[0,1]
    rg = safe_corr(r, g)
    rb = safe_corr(r, b)
    gb = safe_corr(g, b)
    avg = np.mean([rg, rb, gb])
    score = float(np.clip((avg + 1) / 2.0, 0.0, 1.0))
    return score

# =========================
# Combined forensic analyzer
# =========================
def forensic_analyze(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Cannot read image for forensic analysis.")

    freq_s, _ = frequency_score(image)
    noise_s, _ = noise_score(image)
    comp_s, _ = compression_score(image)
    color_s = color_corr_score(image)

    w_freq, w_noise, w_comp, w_color = 0.35, 0.3, 0.2, 0.15
    forensic_score = w_freq * freq_s + w_noise * noise_s + w_comp * comp_s + w_color * color_s
    forensic_score = float(np.clip(forensic_score, 0.0, 1.0))

    return {
        "forensic_score": forensic_score,
        "frequency": float(freq_s),
        "noise": float(noise_s),
        "compression": float(comp_s),
        "color_corr": float(color_s)
    }

# =========================
# Streamlit UI
# =========================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Choose an image file to begin.")
else:
    with open("temp_image.png", "wb") as out:
        out.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    # --- Metadata
    st.subheader("🔎 Local Analysis — Metadata")
    metadata = extract_metadata("temp_image.png")
    if metadata:
        st.success("✅ EXIF metadata found")
        metadata_adjustment = 0
    else:
        st.warning("⚠️ No EXIF metadata found (suspicious)")
        metadata_adjustment = 10

    # --- Forensic
    #st.subheader("🔬 Forensic Analysis")
    forensic = forensic_analyze("temp_image.png")

    # --- Sightengine API
    #st.subheader("🌐 Sightengine API")
    try:
        files = {'media': open("temp_image.png", "rb")}
        data = {'models': 'genai', 'api_user': API_USER, 'api_secret': API_SECRET}
        resp = requests.post("https://api.sightengine.com/1.0/check.json", files=files, data=data, timeout=30)
        api_result = resp.json()
    except Exception as e:
        st.error(f"Sightengine API error: {e}")
        api_result = {}

    api_ai_score = api_result.get("type", {}).get("ai_generated", 0.0) * 100.0 if api_result.get("status") == "success" else 0.0

    # --- Final decision
    forensic_score = forensic['forensic_score']
    forensic_ai_pct = (1.0 - forensic_score) * 100.0
    api_w, local_w = 0.7, 0.3
    local_component = forensic_ai_pct + metadata_adjustment

    final_ai_score = (api_ai_score * api_w) + (local_component * local_w)
    final_ai_score = float(np.clip(final_ai_score, 0.0, 100.0))

    st.subheader("📊 Final Hybrid Decision")
    st.metric("Hybrid AI Likelihood Score", f"{final_ai_score:.2f}%")
    st.progress(int(final_ai_score))

    if final_ai_score > 60:
        st.error("⚠️ This image is likely AI-generated.")
    else:
        st.success("✅ This image is likely real.")

    # --- Contribution Visualization
    # --- Contribution Visualization (Interactive)
   
    
    # --- Pie Chart (Interactive)
        # --- Interactive Pie & Donut Charts (Improved)
    import plotly.graph_objects as go
    import plotly.express as px
    
    st.markdown("---")
    st.subheader("🧠 Contribution Breakdown")
    
    metrics = ["Sightengine", "Frequency", "Noise", "Compression", "Color", "Metadata"]
    scores = [
        api_ai_score,
        (1 - forensic["frequency"]) * 100,
        (1 - forensic["noise"]) * 100,
        (1 - forensic["compression"]) * 100,
        (1 - forensic["color_corr"]) * 100,
        metadata_adjustment
    ]
    
   
    
    # --- Donut Chart with Center Label + Hover + Legend
    donut_fig = go.Figure(
        data=[go.Pie(
            labels=metrics,
            values=scores,
            hole=0.55,
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Contribution: %{percent}<br>Score: %{value:.2f}%<extra></extra>",
            marker=dict(colors=px.colors.qualitative.Pastel)
        )]
    )
    
    donut_fig.update_traces(
        pull=[0.02] * len(metrics),  # slight separation for clarity
        textfont=dict(size=13, color="black")
    )
    
    donut_fig.update_layout(
        title=dict(
            text="🍩 Donut Chart — Weighted Metric Contributions",
            x=0.5,
            xanchor="center"
        ),
        annotations=[dict(
            text=f"{final_ai_score:.1f}%",
            x=0.5, y=0.5,
            font=dict(color="black", family="Arial Black", size=26),
            showarrow=False
        )],
        legend=dict(
            title="Metrics",
            orientation="v",
            x=1.1,
            y=0.95,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        height=600,  # bigger chart height
        margin=dict(t=80, b=50, l=50, r=150),
    )
    
    st.plotly_chart(donut_fig, use_container_width=True)




    # --- Explanation
    st.markdown("---")
    st.subheader("🧩 Explanation")
    st.markdown(f"""
    - **Sightengine (70%)** → {api_ai_score:.1f}% AI likelihood.  
    - **Forensic local (30%)** → AI-likelihood {forensic_ai_pct:.1f}% (score = {forensic_score:.3f}).  
    - **Metadata** → {'No EXIF metadata found (suspicious)' if metadata_adjustment>0 else 'EXIF present (authentic)'}.  
    """)
