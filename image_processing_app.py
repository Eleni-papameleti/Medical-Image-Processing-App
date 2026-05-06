# -*- coding: utf-8 -*-
"""
Medical Image Processing Streamlit App
Chapters: Display Methods | Spatial Filtering | Frequency Filtering | Tomographic Reconstruction
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import signal
import math
import io
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Image Processing",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --navy:       #0a1f3c;
        --navy-mid:   #0f2d54;
        --navy-light: #163d6e;
        --accent:     #2389da;
        --accent-bright: #38a8f5;
        --accent-glow: rgba(35,137,218,0.18);
        --teal:       #0fb8a0;
        --amber:      #f0a500;
        --red:        #e05252;
        --green:      #22c55e;
        --text-light: #e2eef8;
        --text-dim:   #7da8cc;
        --border:     rgba(56,168,245,0.15);
        --card-bg:    rgba(15,45,84,0.7);
        --glass:      rgba(255,255,255,0.04);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
    }

    /* ── App background — dark navy with subtle grid ── */
    .stApp {
        background: var(--navy);
        background-image:
            linear-gradient(rgba(35,137,218,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(35,137,218,0.035) 1px, transparent 1px);
        background-size: 40px 40px;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* ── Header banner ── */
    .top-header {
        background: linear-gradient(135deg, #0e2a50 0%, #133a6a 50%, #1a4d88 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 26px 36px 20px 36px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
        display: flex;
        align-items: center;
        gap: 18px;
        position: relative;
        overflow: hidden;
    }
    .top-header::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 280px; height: 100%;
        background: radial-gradient(ellipse at 80% 50%, rgba(35,137,218,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .top-header-icon {
        font-size: 2.6rem;
        line-height: 1;
        filter: drop-shadow(0 0 12px rgba(56,168,245,0.5));
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 3px 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .subtitle {
        font-size: 0.88rem;
        color: var(--text-dim);
        margin: 0;
        font-weight: 400;
        letter-spacing: 0.2px;
    }
    .header-badges {
        margin-top: 8px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .hbadge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .hbadge-blue  { background: rgba(35,137,218,0.2); color: #7ecef5; border: 1px solid rgba(35,137,218,0.3); }
    .hbadge-teal  { background: rgba(15,184,160,0.18); color: #5fe0cc; border: 1px solid rgba(15,184,160,0.3); }
    .hbadge-amber { background: rgba(240,165,0,0.18); color: #f5c84a; border: 1px solid rgba(240,165,0,0.3); }
    .hbadge-green { background: rgba(34,197,94,0.18); color: #6ee89c; border: 1px solid rgba(34,197,94,0.3); }

    /* ── Chapter header ── */
    .chapter-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-light);
        border-left: 4px solid var(--accent);
        padding: 8px 0 8px 16px;
        margin-bottom: 20px;
        background: linear-gradient(90deg, rgba(35,137,218,0.1) 0%, transparent 100%);
        border-radius: 0 8px 8px 0;
        letter-spacing: -0.1px;
    }

    /* ── Glass cards / section boxes ── */
    .section-box {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 16px;
        color: var(--text-light);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .section-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        background: linear-gradient(180deg, var(--accent), var(--teal));
        border-radius: 4px 0 0 4px;
    }
    .section-box b {
        color: var(--accent-bright);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: block;
        margin-bottom: 8px;
    }
    .section-box i {
        color: #a8d8f0;
        font-style: normal;
        font-weight: 500;
    }

    /* ── Metric chips ── */
    .metric-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 12px 0;
    }
    .metric-chip {
        background: rgba(35,137,218,0.1);
        border: 1px solid rgba(35,137,218,0.22);
        border-radius: 8px;
        padding: 8px 16px;
        text-align: center;
        min-width: 110px;
    }
    .metric-chip .mc-label {
        font-size: 0.7rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        display: block;
        margin-bottom: 2px;
    }
    .metric-chip .mc-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--accent-bright);
        font-family: 'DM Mono', monospace;
    }
    .metric-chip .mc-value.good  { color: #4ade80; }
    .metric-chip .mc-value.warn  { color: var(--amber); }
    .metric-chip .mc-value.bad   { color: var(--red); }

    /* ── Kernel display box ── */
    .kernel-box {
        background: rgba(15,45,84,0.6);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }
    .kernel-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--accent-bright);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    .kernel-table {
        border-collapse: separate;
        border-spacing: 4px;
        margin: 0 auto;
    }
    .kernel-table td {
        width: 50px;
        height: 42px;
        text-align: center;
        vertical-align: middle;
        font-size: 0.95rem;
        font-weight: 700;
        font-family: 'DM Mono', monospace;
        color: var(--text-light);
        border: 1px solid rgba(56,168,245,0.2);
        border-radius: 6px;
        background: rgba(35,137,218,0.12);
    }
    .kernel-table td.kzero {
        color: rgba(125,168,204,0.5);
        background: rgba(15,45,84,0.4);
    }
    .kernel-table td.kneg {
        color: #f87171;
        background: rgba(224,82,82,0.12);
        border-color: rgba(224,82,82,0.25);
    }

    /* ── Info / welcome box ── */
    .welcome-box {
        background: linear-gradient(135deg, rgba(15,45,84,0.9) 0%, rgba(20,60,105,0.9) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
        color: var(--text-light);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .welcome-box h2 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin: 0 0 10px 0;
    }
    .welcome-box ul li {
        line-height: 2;
        color: var(--text-dim);
    }
    .welcome-box ul li b {
        color: var(--text-light);
    }
    .welcome-box hr {
        border: none;
        border-top: 1px solid rgba(56,168,245,0.15);
        margin: 16px 0;
    }

    /* ── No-image info state ── */
    .stAlert {
        background: rgba(35,137,218,0.1) !important;
        border: 1px solid rgba(35,137,218,0.25) !important;
        border-radius: 10px !important;
        color: var(--text-light) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        transition: all 0.18s ease;
        letter-spacing: 0.2px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(35,137,218,0.3);
    }
    .save-btn > button {
        background: linear-gradient(135deg, #0f8c6e, #0fb8a0) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 3px 12px rgba(15,184,160,0.3) !important;
    }
    .save-btn > button:hover {
        box-shadow: 0 6px 20px rgba(15,184,160,0.45) !important;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        border-radius: 9px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #0f8c6e, #0fb8a0) !important;
        color: white !important;
        border: none !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081930 0%, #0a1f3c 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-light) !important;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: var(--text-dim) !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: var(--accent-bright) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(56,168,245,0.12) !important;
        margin: 14px 0 !important;
    }
    /* Sidebar nav pills */
    section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        font-size: 0.88rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--text-dim) !important;
        background: transparent !important;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        border: 1px solid transparent;
        transition: all 0.15s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-light) !important;
        background: rgba(35,137,218,0.08) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-bright) !important;
        background: rgba(35,137,218,0.12) !important;
        border-color: var(--border) !important;
        border-bottom-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: var(--accent) !important;
        height: 2px !important;
    }

    /* ── Selectbox / slider / inputs dark styling ── */
    .stSelectbox [data-baseweb="select"] > div,
    .stTextInput input {
        background: rgba(15,45,84,0.7) !important;
        border-color: var(--border) !important;
        color: var(--text-light) !important;
    }
    .stSlider [data-testid="stSlider"] {
        color: var(--text-dim) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(15,45,84,0.5) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-light) !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: rgba(10,31,60,0.6) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }

    /* ── Footer ── */
    .app-footer {
        margin-top: 3rem;
        padding: 18px 0 8px 0;
        border-top: 1px solid var(--border);
        font-size: 0.76rem;
        color: var(--text-dim);
        text-align: center;
        line-height: 1.8;
    }
    .app-footer a {
        color: var(--accent-bright);
        text-decoration: none;
    }

    /* ── Sidebar nav label ── */
    .sidenav-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #38a8f5;
        margin: 16px 0 4px 0;
        display: block;
        opacity: 0.8;
    }
    .sidebar-credit {
        background: rgba(35,137,218,0.08);
        border: 1px solid rgba(35,137,218,0.15);
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.76rem;
        color: var(--text-dim);
        line-height: 1.7;
        margin-top: 8px;
    }
    .sidebar-credit b {
        color: var(--text-light);
    }
    .sidebar-credit .uni-tag {
        display: inline-block;
        margin-top: 6px;
        padding: 2px 8px;
        background: rgba(35,137,218,0.15);
        border-radius: 4px;
        font-size: 0.7rem;
        color: var(--accent-bright);
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-header">
    <div class="top-header-icon">🔬</div>
    <div class="top-header-text">
        <div class="main-title">Medical Image Processing</div>
        <div class="subtitle">Frequency &amp; Spatial Domain Filtering &nbsp;&middot;&nbsp; Display Methods &nbsp;&middot;&nbsp; Tomographic Reconstruction</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# WELCOME MESSAGE — shown only on first load (no image uploaded yet)
# ─────────────────────────────────────────────────────────────────────────────
if "welcomed" not in st.session_state:
    st.session_state["welcomed"] = True
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 4px 18px rgba(41,128,185,0.18);
    ">
        <h2 style="margin-top:0; color:white; font-size:1.6rem;">👋 Welcome to the Medical Image Processing App!</h2>
        <p style="font-size:1.05rem; margin-bottom:10px;">
            This interactive tool lets you explore and apply a wide range of image processing techniques used in medical imaging.
        </p>
        <hr style="border-color:rgba(255,255,255,0.3); margin:14px 0;">
        <b>🗂️ What you can do:</b>
        <ul style="margin-top:8px; line-height:1.9;">
            <li>📊 <b>Chapter 1</b> — Explore display methods: windowing, LUT transforms, and histogram equalization</li>
            <li>🔲 <b>Chapter 2</b> — Apply spatial domain filters (smoothing, edge detection, sharpening) and view intensity profiles</li>
            <li>〰️ <b>Chapter 3</b> — Frequency domain filtering with Butterworth, Ideal, Gaussian and Wiener restoration</li>
            <li>🔁 <b>Chapter 4</b> — Tomographic reconstruction using FBP and ART algorithms</li>
        </ul>
        <hr style="border-color:rgba(255,255,255,0.3); margin:14px 0;">
        <p style="margin-bottom:0; font-size:0.97rem;">
            👈 <b>To get started:</b> Upload an image (PNG, JPG, or DICOM) from the <b>sidebar</b>, then select a chapter to explore.
            Use the <b>Zoom</b> slider in the sidebar to adjust image display size at any time.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — shared utilities
# ─────────────────────────────────────────────────────────────────────────────
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def imNormalize(w, tones=256):
    mx = np.max(w); mn = np.min(w)
    if mx == mn:
        return np.zeros_like(w)
    w = (tones - 1) * (w - mn) / (mx - mn)
    return np.round(w)

def load_image(uploaded):
    """Load uploaded image (standard or DICOM) as float grayscale array."""
    filename = uploaded.name.lower()
    if filename.endswith(".dcm"):
        try:
            import pydicom
        except ImportError:
            raise RuntimeError("pydicom is not installed. Run: pip install pydicom")
        raw_bytes = uploaded.read()
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            ds = pydicom.dcmread(tmp_path)
            arr = ds.pixel_array.astype(float)
        finally:
            os.unlink(tmp_path)
        if arr.ndim == 3:
            arr = rgb2gray(arr)
        return arr
    else:
        img = Image.open(uploaded).convert("RGB")
        arr = np.array(img, dtype=float)
        if arr.ndim == 3:
            arr = rgb2gray(arr)
        return arr

def get_dicom_metadata(uploaded):
    """Return a dict of key DICOM tags for display."""
    try:
        import pydicom
        raw_bytes = uploaded.read()
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        ds = pydicom.dcmread(tmp_path)
        os.unlink(tmp_path)
        tags = {}
        for tag in ["PatientName", "PatientID", "StudyDate", "Modality",
                    "Rows", "Columns", "PixelSpacing", "SliceThickness",
                    "BitsAllocated", "BitsStored", "RescaleSlope", "RescaleIntercept",
                    "WindowCenter", "WindowWidth", "InstitutionName", "Manufacturer"]:
            val = getattr(ds, tag, None)
            if val is not None:
                tags[tag] = str(val)
        return tags
    except Exception as e:
        return {"Error": str(e)}

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return Image.open(buf).copy()

def download_button(img_array, label="💾 Save Processed Image", key="dl"):
    """Render a PNG download button for a float image array."""
    arr = np.clip(img_array, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    st.download_button(label, data=buf.getvalue(),
                       file_name="processed_image.png",
                       mime="image/png", key=key)

def apply_simple_window(im, wc, ww, image_depth=255, tones=256):
    im1 = np.asarray(im, dtype=float)
    Vb = (2.0 * wc + ww) / 2.0
    Va = Vb - ww
    Vb = min(Vb, image_depth)
    Va = max(Va, 0)
    out = np.zeros_like(im1)
    for idx in np.ndindex(im1.shape):
        v = im1[idx]
        if v < Va:
            out[idx] = 0
        elif v > Vb:
            out[idx] = tones - 1
        else:
            out[idx] = (tones - 1) * (v - Va) / (Vb - Va)
    return np.round(out)

def show_image_row(images, titles, cmap="gray", vmin=0, vmax=255):
    """Display a list of images in columns."""
    cols = st.columns(len(images))
    for col, img, title in zip(cols, images, titles):
        fig, ax = plt.subplots(figsize=(4 * zoom_level, 4 * zoom_level))
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        col.pyplot(fig, use_container_width=True)
        plt.close(fig)

def zoom_fig(w, h):
    """Return a figsize tuple scaled by the current zoom level."""
    return (w * zoom_level, h * zoom_level)

# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 1 FUNCTIONS — Display Methods
# ─────────────────────────────────────────────────────────────────────────────
def linearDisplay(im, image_depth=255, tones=256):
    return np.round((tones - 1) / (image_depth - 1) * np.clip(im, 0, image_depth))

def optimalDisplay(im, tones=256):
    vmn = np.min(im); vmx = np.max(im)
    if vmx == vmn: return np.zeros_like(im)
    return np.round((tones - 1) * (im - vmn) / (vmx - vmn))

def simpleWindow(im, wc, ww, image_depth=255, tones=256):
    return apply_simple_window(im, wc, ww, image_depth, tones)

def brokenWindow(im, image_depth=255, tones=256, gray_val=128, im_val=70):
    im = np.asarray(im, dtype=float)
    im1 = np.zeros_like(im)
    mask1 = im <= im_val
    mask2 = ~mask1
    im1[mask1] = (gray_val / im_val) * im[mask1]
    im1[mask2] = ((tones - 1 - gray_val - 1) / (image_depth - im_val - 1)) * (im[mask2] - im_val - 1) + (gray_val + 1)
    return np.round(np.clip(im1, 0, tones - 1))

def doubleWindow(im, ww1, wl1, ww2, wl2, image_depth=255, tones=256):
    im = np.asarray(im, dtype=float)
    im1 = np.zeros_like(im)
    half = tones / 2 - 1
    ve1 = round((2.0 * wl1 + ww1) / 2.0)
    vs1 = ve1 - ww1
    ve2 = round((2.0 * wl2 + ww2) / 2.0)
    vs2 = ve2 - ww2
    if vs2 < ve1:
        new_point = round((vs2 + ve1) / 2.0)
        ve1 = new_point; vs2 = ve1
    vs1 = max(vs1, 0); ve2 = min(ve2, image_depth)
    for idx in np.ndindex(im.shape):
        v = im[idx]
        if v < vs1:
            im1[idx] = 0
        elif vs1 <= v <= ve1:
            im1[idx] = round((half / (ve1 - vs1)) * (v - vs1)) if ve1 != vs1 else 0
        elif ve1 < v < vs2:
            im1[idx] = half + 1
        elif vs2 <= v <= ve2:
            im1[idx] = round(((tones - 1 - half - 1) / (ve2 - vs2)) * (v - vs2) + (half + 1)) if ve2 != vs2 else half + 1
        else:
            im1[idx] = tones - 1
    return im1

def inverse_lut(tones=256):
    w = np.array([tones - i - 1 for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def logarithmic_lut(tones=256, r=0.05):
    w = np.array([math.log(1 + r * i) for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def inverse_log_lut(tones=256):
    c = tones - 1
    w = np.array([math.exp(i) ** (1 / c) - 1 for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def power_lut(tones=256, gamma=2):
    w = np.array([i ** gamma for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def sine_lut(tones=256):
    w = np.array([np.sin(2 * np.pi * i / 4 * (tones - 1)) for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def exponential_lut(tones=256):
    w = np.array([np.exp(i / 20) for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def sigmoid_lut(tones=256):
    w = np.array([1 / (1 + np.exp(-i / 70)) for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def cosine_lut(tones=256):
    w = np.array([np.cos(2 * np.pi * i / 4 * (tones - 1)) for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def third_lut(tones=256):
    w = np.array([0 if i == 0 else 1 / np.sqrt(i) for i in range(tones)], dtype=float)
    return imNormalize(w, tones)

def apply_lut(im, lut):
    im_int = np.clip(np.round(im), 0, len(lut) - 1).astype(int)
    return lut[im_int]

def f_histogram(A, image_depth=255, tones=256):
    B = A if np.max(A) <= (tones - 1) else np.round((tones - 1) * ((A - 0) / (image_depth - 0)))
    Bval = B.flatten()
    h = np.zeros(tones, dtype=float)
    for v in Bval:
        h[int(np.clip(v, 0, tones - 1))] += 1
    return h

def f_hequalization(A, image_depth=255, tones=256):
    B = np.round((tones - 1) * ((A - 0) / (image_depth - 0)))
    M, N = B.shape
    Bval = B.flatten()
    p = np.argsort(Bval)
    neq = int(M * N / tones + 0.5)
    az = int((M * N) / neq)
    zRem = int((M * N) % neq)
    D = np.zeros(M * N)
    k = -1
    for i in range(0, neq * az, neq):
        k += 1
        D[i:i + neq] = k
    if zRem > 0:
        D[neq * az: neq * az + zRem] = tones - 1
    L = np.zeros(M * N)
    for idx, pi in enumerate(p):
        L[pi] = D[idx]
    Z = L.reshape(B.shape)
    return imNormalize(Z, tones)

def CDF_equalization(img):
    img_u = np.clip(img, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(img_u.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_filled = np.ma.filled(cdf_m, 0).astype("uint8")
    return cdf_filled[img_u].astype(float)

def CLAHE_equalization(img):
    import cv2
    img_u = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_u).astype(float)

def histCumsum(im):
    hist, _ = np.histogram(im.flatten(), 256, [0, 256])
    return hist.cumsum() * hist.max() / hist.cumsum().max()

# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 2 FUNCTIONS — Spatial Filtering
# ─────────────────────────────────────────────────────────────────────────────
def conv2(im, mask):
    return signal.convolve2d(im, mask, mode="same")

def image_convert(im, kernel, image_depth=255):
    sK = np.sum(kernel)
    kernel = kernel.astype(float)
    if sK > 0:
        kernel = kernel / sK
    im2 = conv2(im, kernel)
    return np.clip(im2, 0, image_depth)

# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 3 FUNCTIONS — Frequency Filtering
# ─────────────────────────────────────────────────────────────────────────────
def _lm(N):
    if N % 2 == 0:
        return int(round(N / 2 + 1)), int(round(N / 2 + 2))
    return int(round(N / 2 + 0.5)), int(round(N / 2 + 1 + 0.5))

def _mirror(fh, M, N):
    for k in range(M - 1, N):
        fh[k] = fh[N - k]
    return fh

def ideal_filter(N, fco, TYPE, enh=0.0, trans=0, w=0):
    N = int(N); fco = int(fco); trans = int(trans); w = int(w)
    L, M = _lm(N)
    fh = np.ones(N, dtype=float)
    if TYPE == 1:
        for k in range(fco, L): fh[k] = enh
    elif TYPE == 2:
        fh[:] = enh
        for k in range(fco, L): fh[k] = 1.0
    elif TYPE == 3:
        for k in range(max(0, int(trans - w // 2)), min(N, int(trans + w // 2))): fh[k] = enh
    elif TYPE == 4:
        fh[:] = enh
        for k in range(max(0, int(trans - w // 2)), min(N, int(trans + w // 2))): fh[k] = 1.0
    fh = _mirror(fh, M, N)
    return fh / max(np.max(fh), 1e-10)

def butterworth_filter(N, ndegree, fco, TYPE, trans):
    N = int(N); fco = max(fco, 1); ndegree = max(ndegree, 1)
    L, M = _lm(N)
    fh = np.zeros(N, dtype=float)
    if TYPE == 1:
        for k in range(L): fh[k] = 1.0 / (1.0 + 0.414 * (k / fco) ** (2 * ndegree))
    elif TYPE == 2:
        for k in range(L): fh[k] = 1.0 / (1.0 + 0.414 * (fco / (k + 0.001)) ** (2 * ndegree))
        for k in range(L):
            fh[k] = fh[k + int(trans)] if k < int(N / 2 - trans) else fh[int(N / 2)]
    elif TYPE == 3:
        d = trans
        for k in range(L): fh[k] = 1.0 / (1.0 + 0.414 * (fco / (k - d + 0.001)) ** (2 * ndegree))
    elif TYPE == 4:
        d = trans; fh[:] = 0.001
        for k in range(L): fh[k] = 1.0 / (1.0 + 0.414 * ((k - d) / fco) ** (2 * ndegree))
    fh = _mirror(fh, M, N)
    return fh / max(np.max(fh), 1e-10)

def exponential_filter(N, ndegree, fco, TYPE, trans):
    N = int(N); fco = max(fco, 1); ndegree = max(ndegree, 1)
    L, M = _lm(N)
    fh = np.zeros(N, dtype=float)
    if TYPE == 1:
        for k in range(L): fh[k] = np.exp((-np.log(2)) * (k / fco) ** ndegree)
    elif TYPE == 2:
        for k in range(L): fh[k] = np.exp((-np.log(2)) * (fco / (k + 1e-4)) ** ndegree)
        for k in range(L):
            fh[k] = fh[k + int(trans)] if k < int(N / 2 - trans) else fh[int(N / 2)]
    elif TYPE == 3:
        d = trans
        for k in range(L): fh[k] = np.exp((-np.log(2)) * (fco / (k - d + 1e-5)) ** ndegree)
    elif TYPE == 4:
        d = trans; fh[:] = 0.001
        for k in range(L): fh[k] = np.exp((-np.log(2)) * ((k - d) / fco) ** ndegree)
    fh = _mirror(fh, M, N)
    return fh / max(np.max(fh), 1e-10)

def gaussian_filter_1d(N, ndegree, fco, TYPE, trans):
    N = int(N); fco = max(fco, 1); ndegree = max(ndegree, 1)
    L, M = _lm(N)
    fh = np.zeros(N, dtype=float)
    if TYPE == 1:
        for k in range(L): fh[k] = np.exp(-(k ** 2 / (2 * fco ** 2)) ** ndegree)
    elif TYPE == 2:
        for k in range(L): fh[k] = np.exp(-(2 * fco ** 2 / (k + 1e-4) ** 2) ** ndegree)
        for k in range(L):
            fh[k] = fh[k + int(trans)] if k < int(N / 2 - trans) else fh[int(N / 2)]
    elif TYPE == 3:
        d = trans
        for k in range(L): fh[k] = np.exp(-(2 * fco ** 2 / (k - d + 1e-5) ** 2) ** ndegree)
    elif TYPE == 4:
        d = trans; fh[:] = 0.001
        for k in range(L): fh[k] = np.exp(-((k - d) ** 2 / (2 * fco ** 2)) ** ndegree)
    fh = _mirror(fh, M, N)
    return fh / max(np.max(fh), 1e-10)

def design2dFilter(im, fh):
    y, x = im.shape
    # Build radial distance matrix in one vectorized step
    ky = np.arange(y)
    kx = np.arange(x)
    K = y / 2 - ky[:, None] + 1   # (y, 1)
    M = x / 2 - kx[None, :] + 1   # (1, x)
    ir = np.clip(np.round(np.sqrt(K**2 + M**2)).astype(int), 0, len(fh) - 1)
    FH = fh[ir]
    return np.fft.fftshift(FH)

def filterImage(im, FH):
    Fim = np.fft.fft2(im)
    return np.real(np.fft.ifft2(Fim * FH))

def ampl_fft2(im):
    im1 = np.fft.fftshift(np.fft.fft2(im))
    return np.round(10.0 * np.log(np.abs(im1) + 1))

# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 4 FUNCTIONS — Tomographic Reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def generalizedWienerFilter(fh, filtType, SIGMA):
    N = len(fh)
    C = 2 * SIGMA ** 2
    if filtType == 1: a, b = 1, 0
    elif filtType == 2: a, b = 0, 1
    else: a, b = 0.5, 1
    fhh = np.zeros(N, dtype=float)
    for k in range(N):
        denom = fh[k] ** 2 + b * C
        base = (fh[k] ** 2 / denom) ** (1 - a) if denom > 0 else 0
        inv = 1 / fh[k] if fh[k] > C else 1 / C
        fhh[k] = base * inv
    return fhh / max(np.max(fhh), 1e-10)

def from1dTo2dFilter(im, fh):
    y, x = im.shape
    ky = np.arange(y)
    kx = np.arange(x)
    K = y / 2 - ky[:, None] + 1
    M = x / 2 - kx[None, :] + 1
    ir = np.clip(np.round(np.sqrt(K**2 + M**2)).astype(int), 0, len(fh) - 1)
    FH = fh[ir].astype(float)
    mx = np.amax(FH)
    return FH / mx if mx > 0 else FH

def GaussianMTF(N):
    N = int(N)
    fh = np.zeros(N, dtype=float)
    L, M = _lm(N)
    sigma = L / 2 - 1
    for k in range(L):
        fh[k] = np.exp(-k ** 2 / (2 * sigma ** 2))
    fh = _mirror(fh, M, N)
    return fh

def ST_ERROR(a, b):
    return np.sqrt(np.sum((a - b) ** 2) / len(a))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📂 Navigation")
chapter = st.sidebar.radio(
    "Select Chapter",
    [
        "📊 Chapter 1 — Display Methods",
        "🔲 Chapter 2 — Spatial Filtering",
        "〰️ Chapter 3 — Frequency Filtering",
        "🔁 Chapter 4 — Tomographic Reconstruction",
    ],
)
st.sidebar.markdown("---")

# ── ZOOM SLIDER ──────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🔍 Image Zoom")
zoom_level = st.sidebar.slider(
    "Zoom level",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Scale the display size of all images. Does not affect processing.",
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Upload Image")
upload_mode = st.sidebar.radio("Image type", ["Standard image", "DICOM (.dcm)"], horizontal=True)

if upload_mode == "Standard image":
    uploaded_file = st.sidebar.file_uploader(
        "PNG / JPG / BMP / TIF",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="global_upload_std",
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "DICOM file (.dcm)",
        type=["dcm"],
        key="global_upload_dcm",
    )
    if uploaded_file is not None:
        st.sidebar.markdown("#### 🏥 DICOM Metadata")
        uploaded_file.seek(0)
        meta = get_dicom_metadata(uploaded_file)
        uploaded_file.seek(0)
        priority_tags = ["PatientName", "PatientID", "StudyDate", "Modality",
                         "Rows", "Columns", "BitsStored", "WindowCenter", "WindowWidth",
                         "RescaleSlope", "RescaleIntercept", "PixelSpacing",
                         "SliceThickness", "InstitutionName", "Manufacturer"]
        shown = {t: meta[t] for t in priority_tags if t in meta}
        other = {k: v for k, v in meta.items() if k not in shown and k != "Error"}
        for k, v in shown.items():
            st.sidebar.markdown(f"<small><b>{k}:</b> {v}</small>", unsafe_allow_html=True)
        if other:
            with st.sidebar.expander("More tags…"):
                for k, v in other.items():
                    st.sidebar.markdown(f"<small><b>{k}:</b> {v}</small>", unsafe_allow_html=True)
        if "Error" in meta:
            st.sidebar.warning(f"Metadata error: {meta['Error']}")

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE WINDOW SIDEBAR — available in all chapters
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🪟 Simple Window Display")
use_window = st.sidebar.checkbox("Apply Simple Window to output", value=False)
if use_window:
    sw_wc = st.sidebar.slider("Window Center (wc)", 0, 255, 128)
    sw_ww = st.sidebar.slider("Window Width (ww)", 1, 512, 256)

def maybe_window(im):
    if use_window:
        return apply_simple_window(im, sw_wc, sw_ww)
    return im

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  CHAPTER 1  ═══════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
if chapter.startswith("📊"):
    st.markdown('<div class="chapter-header">Chapter 1 — Image Display Modification Methods</div>',
                unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("👈 Upload an image from the sidebar to get started.")
        st.stop()

    im_raw = load_image(uploaded_file)
    image_depth = 255; tones = 256

    tab1, tab2 = st.tabs(["🖼️ Window & LUT Methods", "📈 Histogram Equalization"])

    # ── TAB 1: Window / LUT ──────────────────────────────────────────────────
    with tab1:
        col_l, col_r = st.columns([1, 2])
        with col_l:
            method = st.selectbox("Display Method", [
                "Linear Display",
                "Optimal Display",
                "Simple Window",
                "Broken Window",
                "Double Window",
                "Inverse",
                "Logarithmic",
                "Inverse Logarithmic",
                "Power (Gamma)",
                "Sine",
                "Exponential",
                "Sigmoid",
                "Cosine",
                "Third (1/√i)",
            ])

            result = None
            if method == "Linear Display":
                result = linearDisplay(im_raw, image_depth, tones)

            elif method == "Optimal Display":
                result = optimalDisplay(im_raw, tones)

            elif method == "Simple Window":
                wc = st.slider("Window Center", 0, 255, 128)
                ww = st.slider("Window Width", 1, 512, 256)
                result = simpleWindow(im_raw, wc, ww, image_depth, tones)

            elif method == "Broken Window":
                gray_val = st.slider("Gray Value (breakpoint output)", 0, 255, 128)
                im_val = st.slider("Image Value (breakpoint input)", 0, 255, 70)
                result = brokenWindow(im_raw, image_depth, tones, gray_val, im_val)

            elif method == "Double Window":
                wl1 = st.slider("Window Level 1", 0, 255, 50)
                ww1 = st.slider("Window Width 1", 1, 255, 100)
                wl2 = st.slider("Window Level 2", 0, 255, 150)
                ww2 = st.slider("Window Width 2", 1, 255, 100)
                result = doubleWindow(im_raw, ww1, wl1, ww2, wl2, image_depth, tones)

            elif method == "Inverse":
                lut = inverse_lut(tones); result = apply_lut(im_raw, lut)
            elif method == "Logarithmic":
                r = st.slider("Log rate r", 0.001, 0.5, 0.05, step=0.001)
                lut = logarithmic_lut(tones, r); result = apply_lut(im_raw, lut)
            elif method == "Inverse Logarithmic":
                lut = inverse_log_lut(tones); result = apply_lut(im_raw, lut)
            elif method == "Power (Gamma)":
                gamma = st.slider("Gamma", 0.1, 5.0, 2.0, step=0.1)
                lut = power_lut(tones, gamma); result = apply_lut(im_raw, lut)
            elif method == "Sine":
                lut = sine_lut(tones); result = apply_lut(im_raw, lut)
            elif method == "Exponential":
                lut = exponential_lut(tones); result = apply_lut(im_raw, lut)
            elif method == "Sigmoid":
                lut = sigmoid_lut(tones); result = apply_lut(im_raw, lut)
            elif method == "Cosine":
                lut = cosine_lut(tones); result = apply_lut(im_raw, lut)
            elif method == "Third (1/√i)":
                lut = third_lut(tones); result = apply_lut(im_raw, lut)

        with col_r:
            if result is not None:
                result_win = maybe_window(result)
                fig, axes = plt.subplots(1, 3, figsize=zoom_fig(12, 4))
                axes[0].imshow(im_raw, cmap="gray", vmin=0, vmax=255)
                axes[0].set_title("Original Image"); axes[0].axis("off")
                axes[1].imshow(result, cmap="gray", vmin=0, vmax=255)
                axes[1].set_title(method); axes[1].axis("off")
                axes[2].imshow(result_win, cmap="gray", vmin=0, vmax=255)
                axes[2].set_title("After Simple Window" if use_window else "Output (no window)")
                axes[2].axis("off")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # LUT plot for LUT methods
                lut_methods = ["Inverse","Logarithmic","Inverse Logarithmic","Power (Gamma)","Sine","Exponential","Sigmoid","Cosine","Third (1/√i)"]
                if method in lut_methods:
                    fig2, ax2 = plt.subplots(figsize=zoom_fig(5, 2.5))
                    ax2.plot(lut, color="#2980b9")
                    ax2.set_title("LUT Curve"); ax2.set_xlabel("Input"); ax2.set_ylabel("Output")
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                st.markdown('<div class="save-btn">', unsafe_allow_html=True)
                download_button(result_win, "💾 Save Processed Image", key="ch1_save")
                st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 2: Histogram Equalization ─────────────────────────────────────────
    with tab2:
        col_l2, col_r2 = st.columns([1, 2])
        with col_l2:
            eq_method = st.selectbox("Equalization Method", [
                "Histogram Equalization (custom)",
                "CDF Equalization (OpenCV)",
                "CLAHE (OpenCV)",
            ])
        with col_r2:
            im_u = np.clip(im_raw, 0, 255)
            if eq_method == "Histogram Equalization (custom)":
                eq_result = f_hequalization(im_u, 255, 256)
            elif eq_method == "CDF Equalization (OpenCV)":
                eq_result = CDF_equalization(im_u)
            else:
                eq_result = CLAHE_equalization(im_u)

            eq_win = maybe_window(eq_result)

            h_orig = f_histogram(im_u, 255, 256)
            h_eq = f_histogram(eq_result, 255, 256)
            cs_orig = histCumsum(im_u)
            cs_eq = histCumsum(eq_result)

            fig, axes = plt.subplots(2, 2, figsize=zoom_fig(10, 7))
            axes[0, 0].imshow(im_u, cmap="gray", vmin=0, vmax=255)
            axes[0, 0].set_title("Original Image"); axes[0, 0].axis("off")
            axes[0, 1].imshow(eq_win, cmap="gray", vmin=0, vmax=255)
            axes[0, 1].set_title(f"Equalized ({eq_method})" + (" + Window" if use_window else ""))
            axes[0, 1].axis("off")
            axes[1, 0].bar(range(256), h_orig, color="#2980b9", alpha=0.7, width=1)
            axes[1, 0].plot(cs_orig, "r", linewidth=1.5); axes[1, 0].set_title("Histogram — Original")
            axes[1, 0].legend(["CDF", "Histogram"]); axes[1, 0].grid(alpha=0.3)
            axes[1, 1].bar(range(256), h_eq, color="#27ae60", alpha=0.7, width=1)
            axes[1, 1].plot(cs_eq, "r", linewidth=1.5); axes[1, 1].set_title("Histogram — Equalized")
            axes[1, 1].legend(["CDF", "Histogram"]); axes[1, 1].grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown('<div class="save-btn">', unsafe_allow_html=True)
            download_button(eq_win, "💾 Save Equalized Image", key="ch1_eq_save")
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  CHAPTER 2  ═══════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif chapter.startswith("🔲"):
    st.markdown('<div class="chapter-header">Chapter 2 — Spatial Domain Image Filtering</div>',
                unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("👈 Upload an image from the sidebar to get started.")
        st.stop()

    im_raw = load_image(uploaded_file)
    image_depth = 255; tones = 256
    im_norm = imNormalize(im_raw, tones)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        mask_type = st.selectbox("Filter / Mask", [
            "Smoothing (cross)",
            "Smoothing (3×3 box)",
            "Smoothing (5×5 box)",
            "Median (3×3)",
            "Median (5×5)",
            "Laplacian",
            "High-Emphasis (sharpening)",
            "Sobel (edge detect)",
        ])
        custom_kernel = st.checkbox("Use custom kernel")
        if custom_kernel:
            st.markdown("Enter a 3×3 kernel (space-separated rows):")
            row1 = st.text_input("Row 1", "0 1 0")
            row2 = st.text_input("Row 2", "1 1 1")
            row3 = st.text_input("Row 3", "0 1 0")
            try:
                kernel = np.array([[float(x) for x in r.split()] for r in [row1, row2, row3]])
            except:
                st.error("Invalid kernel — using identity.")
                kernel = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float)

    with col_r:
        if not custom_kernel:
            if mask_type == "Smoothing (cross)":
                kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=float)
                im3 = image_convert(im_norm, kernel, image_depth)
            elif mask_type == "Smoothing (3×3 box)":
                kernel = np.ones((3,3), dtype=float)
                im3 = image_convert(im_norm, kernel, image_depth)
            elif mask_type == "Smoothing (5×5 box)":
                kernel = np.ones((5,5), dtype=float)
                im3 = image_convert(im_norm, kernel, image_depth)
            elif mask_type.startswith("Median (3"):
                im3 = signal.medfilt2d(im_norm.astype(float), (3, 3))
            elif mask_type.startswith("Median (5"):
                im3 = signal.medfilt2d(im_norm.astype(float), (5, 5))
            elif mask_type == "Laplacian":
                kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
                im3 = image_convert(im_norm, kernel, image_depth)
            elif mask_type == "High-Emphasis (sharpening)":
                kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float)
                im3 = image_convert(im_norm, kernel, image_depth)
            elif mask_type == "Sobel (edge detect)":
                kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
                ky = kx.T
                gx = conv2(im_norm, kx); gy = conv2(im_norm, ky)
                im3 = np.sqrt(gx**2 + gy**2)
        else:
            im3 = image_convert(im_norm, kernel, image_depth)

        im3 = imNormalize(im3, tones)

        # Noise analysis
        M, N = im_norm.shape
        c_orig = im_norm[1:M-1, 1:N-1]
        c_filt = im3[1:M-1, 1:N-1]
        std_orig = np.std(c_orig, ddof=1)
        std_filt = np.std(c_filt, ddof=1)
        noise_diff = 100 * (std_orig - std_filt) / (std_orig + 1e-9)
        filter_type_label = "High-Pass (noise increased)" if noise_diff < 0 else "Low-Pass (noise reduced)"

        im3_win = maybe_window(im3)

        fig, axes = plt.subplots(1, 3, figsize=zoom_fig(12, 4))
        axes[0].imshow(im_norm, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(im3, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"Filtered: {mask_type}"); axes[1].axis("off")
        axes[2].imshow(im3_win, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("After Simple Window" if use_window else "Output"); axes[2].axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Intensity Profile Plots ──────────────────────────────────────────
        st.markdown("#### 📈 Pixel Intensity Profiles")
        mid_row = im_norm.shape[0] // 2
        mid_col = im_norm.shape[1] // 2

        fig_p, axes_p = plt.subplots(1, 2, figsize=zoom_fig(12, 3))

        # Horizontal profile (middle row)
        axes_p[0].plot(im_norm[mid_row, :], color="#2980b9", linewidth=1.2, label="Original")
        axes_p[0].plot(im3[mid_row, :], color="#e74c3c", linewidth=1.2, linestyle="--", label="Filtered")
        axes_p[0].set_title(f"Horizontal Profile (row {mid_row})")
        axes_p[0].set_xlabel("Column index")
        axes_p[0].set_ylabel("Pixel Intensity")
        axes_p[0].legend()
        axes_p[0].grid(True, alpha=0.3)
        axes_p[0].set_ylim(0, 255)

        # Vertical profile (middle column)
        axes_p[1].plot(im_norm[:, mid_col], color="#2980b9", linewidth=1.2, label="Original")
        axes_p[1].plot(im3[:, mid_col], color="#e74c3c", linewidth=1.2, linestyle="--", label="Filtered")
        axes_p[1].set_title(f"Vertical Profile (col {mid_col})")
        axes_p[1].set_xlabel("Row index")
        axes_p[1].set_ylabel("Pixel Intensity")
        axes_p[1].legend()
        axes_p[1].grid(True, alpha=0.3)
        axes_p[1].set_ylim(0, 255)

        plt.tight_layout()
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)

        # ── Noise Analysis Box ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="section-box">
        <b>📊 Noise Analysis</b><br><br>
        σ&nbsp;Original &nbsp;= &nbsp;<b style="color:#ffffff;font-size:1.05rem">{std_orig:.3f}</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        σ&nbsp;Filtered &nbsp;= &nbsp;<b style="color:#ffffff;font-size:1.05rem">{std_filt:.3f}</b>
        <br><br>
        Noise Δ = <b style="color:#7ec8f4;font-size:1.08rem">{noise_diff:.2f}%</b>
        &nbsp;→&nbsp; <i>{filter_type_label}</i>
        </div>
        """, unsafe_allow_html=True)

        # ── Filter Kernel Display ───────────────────────────────────────────
        kernel_display = None
        if not custom_kernel:
            if mask_type == "Smoothing (cross)":
                kernel_display = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=float)
            elif mask_type == "Smoothing (3×3 box)":
                kernel_display = np.ones((3,3), dtype=float)
            elif mask_type == "Smoothing (5×5 box)":
                kernel_display = np.ones((5,5), dtype=float)
            elif mask_type.startswith("Median (3"):
                kernel_display = np.ones((3,3), dtype=float)
            elif mask_type.startswith("Median (5"):
                kernel_display = np.ones((5,5), dtype=float)
            elif mask_type == "Laplacian":
                kernel_display = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
            elif mask_type == "High-Emphasis (sharpening)":
                kernel_display = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float)
            elif mask_type == "Sobel (edge detect)":
                kernel_display = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
        else:
            kernel_display = kernel

        if kernel_display is not None:
            rows, cols = kernel_display.shape
            cells = ""
            for r in range(rows):
                cells += "<tr>"
                for c in range(cols):
                    v = kernel_display[r, c]
                    cls = "kzero" if v == 0 else ("kneg" if v < 0 else "")
                    display_val = int(v) if v == int(v) else f"{v:.2f}"
                    cells += f'<td class="{cls}">{display_val}</td>'
                cells += "</tr>"
            label = "Sobel Gx Kernel" if mask_type == "Sobel (edge detect)" else f"{mask_type} Kernel"
            if mask_type.startswith("Median"):
                label = f"{mask_type} — Neighbourhood (equal weights)"
            st.markdown(f"""
            <div class="kernel-box">
                <div class="kernel-title">🔲 Filter Kernel — {label}</div>
                <table class="kernel-table">{cells}</table>
            </div>
            """, unsafe_allow_html=True)
            if mask_type == "Sobel (edge detect)":
                ky_display = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float)
                cells_y = ""
                for r in range(3):
                    cells_y += "<tr>"
                    for c in range(3):
                        v = ky_display[r, c]
                        cls = "kzero" if v == 0 else ("kneg" if v < 0 else "")
                        display_val = int(v)
                        cells_y += f'<td class="{cls}">{display_val}</td>'
                    cells_y += "</tr>"
                st.markdown(f"""
                <div class="kernel-box">
                    <div class="kernel-title">🔲 Filter Kernel — Sobel Gy Kernel</div>
                    <table class="kernel-table">{cells_y}</table>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="save-btn">', unsafe_allow_html=True)
        download_button(im3_win, "💾 Save Filtered Image", key="ch2_save")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  CHAPTER 3  ═══════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif chapter.startswith("〰️"):
    st.markdown('<div class="chapter-header">Chapter 3 — Frequency Domain Image Filtering</div>',
                unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("👈 Upload an image from the sidebar to get started.")
        st.stop()

    im_raw = load_image(uploaded_file)
    tones = 256
    im = imNormalize(im_raw, tones)
    M, N = im.shape
    Flength = int(round(np.sqrt(M * M + N * N)))

    tab_filt, tab_rest = st.tabs(["🔵 Frequency Filtering", "🔧 Image Restoration (Wiener)"])

    # ── TAB: Frequency Filtering ─────────────────────────────────────────────
    with tab_filt:
        col_l, col_r = st.columns([1, 2])
        with col_l:
            filter_family = st.selectbox("Filter Family", ["Butterworth", "Ideal", "Exponential", "Gaussian"])
            filter_type = st.selectbox("Filter Type", ["Low-Pass (LP)", "High-Pass (HP)", "Band-Reject (BR)", "Band-Pass (BP)"])
            type_idx = ["Low-Pass (LP)", "High-Pass (HP)", "Band-Reject (BR)", "Band-Pass (BP)"].index(filter_type) + 1

            ndegree = st.slider("Filter Degree / Order", 1, 10, 2)
            fco_pct = st.slider("Cut-off Frequency (% of max)", 1, 80, 30)
            fco = int(round(Flength * fco_pct / 100))

            if type_idx in [3, 4]:  # BR / BP
                trans_pct = st.slider("Band Center (% of max)", 1, 80, 20)
                w_pct = st.slider("Band Width (% of max)", 1, 60, 15)
                trans = int(round(Flength * trans_pct / 100))
                w = int(round(Flength * w_pct / 100))
            else:
                trans_pct = st.slider("HP shift / trans (% of max)", 0, 50, 25)
                trans = int(round(Flength * trans_pct / 100))
                w = 0

        with col_r:
            with st.spinner("Applying frequency filter…"):
                if filter_family == "Butterworth":
                    fh = butterworth_filter(Flength, ndegree, fco, type_idx, trans)
                elif filter_family == "Ideal":
                    fh = ideal_filter(Flength, fco, type_idx, 0.0, trans, w)
                elif filter_family == "Exponential":
                    fh = exponential_filter(Flength, ndegree, fco, type_idx, trans)
                else:
                    fh = gaussian_filter_1d(Flength, ndegree, fco, type_idx, trans)

                FH = design2dFilter(im, fh)
                im_filt = filterImage(im, FH)
                im_filt = imNormalize(im_filt, tones)
                im_filt_win = maybe_window(im_filt)

                ampl_orig = ampl_fft2(im)
                ampl_filt = ampl_fft2(im_filt)

            fig, axes = plt.subplots(2, 3, figsize=zoom_fig(13, 8))
            axes[0, 0].imshow(im, cmap="gray", vmin=0, vmax=255); axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
            axes[0, 1].imshow(im_filt, cmap="gray", vmin=0, vmax=255); axes[0, 1].set_title(f"{filter_family} {filter_type}"); axes[0, 1].axis("off")
            axes[0, 2].imshow(im_filt_win, cmap="gray", vmin=0, vmax=255)
            axes[0, 2].set_title("After Simple Window" if use_window else "Output"); axes[0, 2].axis("off")
            axes[1, 0].imshow(ampl_orig, cmap="jet"); axes[1, 0].set_title("FFT Spectrum — Original"); axes[1, 0].axis("off")
            axes[1, 1].imshow(ampl_filt, cmap="jet"); axes[1, 1].set_title("FFT Spectrum — Filtered"); axes[1, 1].axis("off")
            axes[1, 2].plot(fh, color="#2980b9"); axes[1, 2].set_title("1D Filter Profile")
            axes[1, 2].set_xlabel("Spatial Frequency"); axes[1, 2].set_ylabel("Amplitude")
            axes[1, 2].set_xlim(0, Flength); axes[1, 2].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown('<div class="save-btn">', unsafe_allow_html=True)
            download_button(im_filt_win, "💾 Save Filtered Image", key="ch3_filt_save")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB: Image Restoration ───────────────────────────────────────────────
    with tab_rest:
        st.markdown("Apply Gaussian blur + noise, then restore using Inverse / Wiener / Power filters.")

        col_l2, col_r2 = st.columns([1, 2])
        with col_l2:
            sigma_noise = st.slider("Noise σ (for restoration)", 0.01, 1.0, 0.25, step=0.01)
            noise_pct = st.slider("% Noise added to image", 0, 50, 10)
            wc_rest = st.slider("Window Center (restoration display)", 0, 255, 130)
            ww_rest = st.slider("Window Width (restoration display)", 1, 512, 256)

        with col_r2:
            with st.spinner("Running restoration pipeline…"):
                # Build Gaussian MTF blur filter
                fh_blur = GaussianMTF(Flength)
                FH_blur = design2dFilter(im, fh_blur)
                # Blur image
                im_blurred = np.real(np.fft.ifft2(np.fft.fft2(im) * np.fft.fftshift(FH_blur)))
                im_blurred = imNormalize(im_blurred, tones)
                # Add noise
                im_noisy = im_blurred.copy()
                im_noisy += noise_pct * im_blurred * np.random.rand(*im_blurred.shape) / 100
                im_noisy = imNormalize(im_noisy, tones)

                diag_orig = np.diag(im[:min(M, N), :min(M, N)])

                results = []
                labels = ["Inverse Filter", "Wiener Filter", "Power Filter"]
                for ft in [1, 2, 3]:
                    fhh = generalizedWienerFilter(fh_blur, ft, sigma_noise)
                    FHH = from1dTo2dFilter(im_noisy, fhh)
                    im_rest = np.real(np.fft.ifft2(np.fft.fft2(im_noisy) * np.fft.fftshift(FHH)))
                    im_rest = imNormalize(im_rest, tones)
                    im_rest_w = apply_simple_window(im_rest, wc_rest, ww_rest)
                    diag_rest = np.diag(im_rest[:min(M, N), :min(M, N)])
                    se = ST_ERROR(diag_orig, diag_rest)
                    results.append((im_rest_w, fhh, se, labels[ft - 1]))

            fig, axes = plt.subplots(3, 3, figsize=zoom_fig(13, 11))
            axes[0, 0].imshow(im, cmap="gray", vmin=0, vmax=255); axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
            axes[0, 1].imshow(im_blurred, cmap="gray", vmin=0, vmax=255); axes[0, 1].set_title("Blurred"); axes[0, 1].axis("off")
            axes[0, 2].imshow(im_noisy, cmap="gray", vmin=0, vmax=255); axes[0, 2].set_title(f"Blurred + {noise_pct}% Noise"); axes[0, 2].axis("off")

            for i, (im_r, fhh_r, se_r, lbl) in enumerate(results):
                axes[1, i].imshow(im_r, cmap="gray", vmin=0, vmax=255)
                axes[1, i].set_title(f"{lbl}\nST_ERR={se_r:.2f}"); axes[1, i].axis("off")
                axes[2, i].plot(fhh_r, color=["red","blue","green"][i])
                axes[2, i].set_title(f"{lbl} Profile"); axes[2, i].set_xlim(0, Flength); axes[2, i].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Save best (Wiener)
            wiener_img = results[1][0]
            wiener_win = maybe_window(wiener_img)
            st.markdown('<div class="save-btn">', unsafe_allow_html=True)
            download_button(wiener_win, "💾 Save Wiener-Restored Image", key="ch3_rest_save")
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════  CHAPTER 4  ═══════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif chapter.startswith("🔁"):
    st.markdown('<div class="chapter-header">Chapter 4 — Tomographic Image Reconstruction Methods</div>',
                unsafe_allow_html=True)

    try:
        from skimage.transform import radon, iradon, iradon_sart
    except ImportError:
        st.error("scikit-image is required for Chapter 4. Install it with: pip install scikit-image")
        st.stop()

    if uploaded_file is None:
        st.info("👈 Upload an image from the sidebar to get started.")
        st.stop()

    im_raw = load_image(uploaded_file)
    # Make square for radon
    s = min(im_raw.shape)
    im_sq = im_raw[:s, :s]
    mx = np.max(im_sq); mn = np.min(im_sq)
    A_full = (im_sq - mn) * (255 / (mx - mn + 1e-9))

    col_l, col_r = st.columns([1, 2])
    with col_l:
        # Downscale option to keep processing fast
        max_size = st.select_slider(
            "Max image size (px)",
            options=[128, 192, 256, 320, 512],
            value=256,
            help="Smaller = much faster. Radon/ART scale as O(n²) per projection.",
        )
        N_proj = st.slider("Number of projections (angles)", 10, 180, 90,
                           help="Fewer projections = faster. 90 is a good balance.")
        fbp_filter = st.selectbox("FBP Filter", ["ramp", "shepp-logan", "cosine", "hamming", "hann", "None"])
        art_iters = st.slider("ART Iterations", 1, 10, 3,
                              help="Each iteration adds processing time.")
        run_btn = st.button("▶ Run Reconstruction")

    with col_r:
        if run_btn:
            with st.spinner("Computing sinogram and reconstructions…"):
                # ── Downsample image if needed ──────────────────────────────
                if s > max_size:
                    from skimage.transform import resize
                    A = resize(A_full, (max_size, max_size),
                               anti_aliasing=True, preserve_range=True)
                else:
                    A = A_full

                # ── Sinogram (cached via session_state key) ─────────────────
                cache_key = f"ch4_sinogram_{id(uploaded_file)}_{N_proj}_{A.shape[0]}"
                if cache_key not in st.session_state:
                    theta = np.linspace(0, 180, N_proj, endpoint=False)
                    st.session_state[cache_key] = (radon(A, theta=theta, circle=True), theta)
                sinogram, theta = st.session_state[cache_key]

                # ── FBP ─────────────────────────────────────────────────────
                filter_name = None if fbp_filter == "None" else fbp_filter
                I_FBP = iradon(sinogram, theta=theta,
                               filter_name=filter_name,
                               circle=True)

                # ── ART (SART) ───────────────────────────────────────────────
                I_ART = iradon_sart(sinogram, theta=theta)
                for _ in range(art_iters - 1):
                    I_ART = iradon_sart(sinogram, theta=theta, image=I_ART)

                I_FBP_n = imNormalize(I_FBP, 256)
                I_ART_n = imNormalize(I_ART, 256)
                I_FBP_win = maybe_window(I_FBP_n)
                I_ART_win = maybe_window(I_ART_n)

            fig, axes = plt.subplots(2, 2, figsize=zoom_fig(12, 10))
            axes[0, 0].imshow(A, cmap="gray"); axes[0, 0].set_title(f"Original ({A.shape[0]}×{A.shape[1]})"); axes[0, 0].axis("off")
            axes[0, 1].imshow(sinogram, cmap="gray",
                              extent=(theta[0], theta[-1], 0, sinogram.shape[0]), aspect="auto")
            axes[0, 1].set_title(f"Sinogram ({N_proj} projections)")
            axes[0, 1].set_xlabel("Angle (°)"); axes[0, 1].set_ylabel("Detector")
            axes[1, 0].imshow(I_FBP_win, cmap="gray")
            axes[1, 0].set_title(f"FBP Reconstruction\nFilter: {fbp_filter}" + (" + Window" if use_window else ""))
            axes[1, 0].axis("off")
            axes[1, 1].imshow(I_ART_win, cmap="gray")
            axes[1, 1].set_title(f"ART Reconstruction ({art_iters} iterations)" + (" + Window" if use_window else ""))
            axes[1, 1].axis("off")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Diagonal profiles
            fig2, axes2 = plt.subplots(1, 2, figsize=zoom_fig(12, 3))
            d_orig = np.diag(A[:min(A.shape), :min(A.shape)])
            d_fbp  = np.diag(I_FBP_n[:min(I_FBP_n.shape), :min(I_FBP_n.shape)])
            d_art  = np.diag(I_ART_n[:min(I_ART_n.shape), :min(I_ART_n.shape)])
            axes2[0].plot(d_orig, "b", label="Original"); axes2[0].plot(d_fbp, "r--", label="FBP")
            axes2[0].set_title("Diagonal Profile — FBP vs Original"); axes2[0].legend(); axes2[0].grid(alpha=0.3)
            axes2[1].plot(d_orig, "b", label="Original"); axes2[1].plot(d_art, "g--", label="ART")
            axes2[1].set_title("Diagonal Profile — ART vs Original"); axes2[1].legend(); axes2[1].grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            col_save1, col_save2 = st.columns(2)
            with col_save1:
                st.markdown('<div class="save-btn">', unsafe_allow_html=True)
                download_button(I_FBP_win, "💾 Save FBP Image", key="ch4_fbp_save")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_save2:
                st.markdown('<div class="save-btn">', unsafe_allow_html=True)
                download_button(I_ART_win, "💾 Save ART Image", key="ch4_art_save")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Configure parameters above and click **▶ Run Reconstruction**.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <small>Medical Image Processing App - Built with Streamlit</small><br>
    <small>Created by: Eleni Papameleti</small><br>
    <small>Project for Image Processing Lab - Biomedical Engineering Department - University of West Attica</small>
    """, 
    unsafe_allow_html=True
   )

