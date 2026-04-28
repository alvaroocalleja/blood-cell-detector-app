"""
Blood Cell Detector — Streamlit Web Application
=================================================
Upload a peripheral blood smear image and get instant AI-powered
cell detection and classification using a YOLO26 model fine-tuned
on 7 cell types: RBC, Platelets, Neutrophil, Lymphocyte, Monocyte,
Eosinophil, and Basophil.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "blood_detector_model.pt"
METADATA_PATH = ROOT / "blood_detector_metadata.json"
TEST_IMAGES_DIR = ROOT / "test_images"

# ── Class colours (RGB for Streamlit / PIL) ──────────────────────────────
CLASS_COLORS = {
    "RBC":        (220, 60, 60),     # soft red
    "Platelets":  (46, 204, 113),    # emerald green
    "Neutrophil": (52, 152, 219),    # steel blue
    "Lymphocyte": (155, 89, 182),    # amethyst purple
    "Monocyte":   (243, 156, 18),    # sunflower orange
    "Eosinophil": (231, 76, 60),     # alizarin red-orange
    "Basophil":   (26, 188, 156),    # turquoise
}

# Hex versions for the UI legend
CLASS_HEX = {k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in CLASS_COLORS.items()}

CLASS_DESCRIPTIONS = {
    "RBC":        "Red Blood Cells — oxygen transport",
    "Platelets":  "Platelets — clotting and hemostasis",
    "Neutrophil": "Neutrophil — most common WBC, fights bacteria",
    "Lymphocyte": "Lymphocyte — adaptive immunity (T & B cells)",
    "Monocyte":   "Monocyte — largest WBC, becomes macrophage",
    "Eosinophil": "Eosinophil — combats parasites & allergies",
    "Basophil":   "Basophil — rarest WBC, mediates inflammation",
}


# ── Helper functions ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading YOLO model …")
def load_model():
    """Load the YOLO model once and cache it across reruns."""
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    return model


def annotate_image(
    img_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    names: dict,
    show_labels: bool = True,
    show_conf: bool = True,
    line_width: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an RGB image and return the annotated copy."""
    annotated = img_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes_xyxy, classes, confidences):
        name = names[int(cls_id)]
        color_rgb = CLASS_COLORS.get(name, (200, 200, 200))
        # OpenCV uses BGR
        color_bgr = color_rgb[::-1]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, line_width)

        if show_labels:
            label = name
            if show_conf:
                label += f" {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y = y1 - 4
            if label_y - th - 4 < 0:
                label_y = y2 + th + 6
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 4),
                (x1 + tw + 6, label_y + 2),
                color_bgr,
                -1,
            )
            cv2.putText(
                annotated, label, (x1 + 3, label_y - 1),
                font, font_scale, (255, 255, 255),
                font_thickness, cv2.LINE_AA,
            )

    # Convert back from BGR to RGB for display
    annotated = cv2.cvtColor(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
    return annotated


def build_summary_table(classes: np.ndarray, names: dict) -> dict:
    """Return a {class_name: count} dict sorted by count descending."""
    counts = Counter(names[int(c)] for c in classes)
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to bytes for download."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Cell Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google font ── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }

    /* ── Custom Scrollbar ── */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #818cf8; }

    /* ── Hero banner ── */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 20px 50px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, transparent 70%);
        filter: blur(40px);
    }
    .hero-container h1 {
        color: #fff;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .hero-container p {
        color: rgba(255,255,255,0.7);
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
        max-width: 800px;
    }
    .hero-container .accent {
        color: #818cf8;
        font-weight: 700;
    }

    /* ── Stat cards ── */
    .stat-row {
        display: flex;
        gap: 1.2rem;
        margin-top: 2rem;
        flex-wrap: wrap;
    }
    .stat-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1rem 1.5rem;
        min-width: 140px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        background: rgba(255,255,255,0.12);
        border-color: rgba(255,255,255,0.2);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .stat-card .stat-value {
        color: #fff;
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
    }
    .stat-card .stat-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    /* ── Legend chips ── */
    .legend-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.8rem;
    }
    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
        transition: all 0.2s ease;
    }
    .legend-chip:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(255,255,255,0.15);
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
        box-shadow: 0 0 10px currentColor;
    }

    /* ── Result cards ── */
    .result-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .result-card h3 {
        color: #f8fafc;
        margin-bottom: 1rem;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ── Count bar ── */
    .count-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.8rem;
        gap: 0.8rem;
    }
    .count-label {
        min-width: 100px;
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .count-bar-bg {
        flex: 1;
        background: #0f172a;
        border-radius: 8px;
        height: 24px;
        overflow: hidden;
    }
    .count-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #fff;
        transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    /* ── Buttons ── */
    .stDownloadButton button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(79, 70, 229, 0.4) !important;
        filter: brightness(1.1) !important;
    }

    /* ── Hide default elements ── */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] section {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.markdown("---")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Minimum confidence score to keep a detection. Lower = more detections (possibly noisier).",
    )
    iou_threshold = st.slider(
        "IoU threshold (NMS)",
        min_value=0.1, max_value=0.95, value=0.7, step=0.05,
        help="Non-max suppression overlap threshold. Higher = allows more overlapping boxes.",
    )
    img_size = st.select_slider(
        "Inference resolution",
        options=[320, 480, 640, 800, 1024],
        value=640,
        help="Image size fed to the model. 640 matches training resolution.",
    )

    st.markdown("---")
    st.markdown("## 🎨 Display Options")
    show_labels = st.checkbox("Show class labels", value=True)
    show_conf = st.checkbox("Show confidence scores", value=True)
    line_width = st.slider("Box line width", 1, 5, 2)

    st.markdown("---")
    st.markdown("## 📖 Class Legend")
    legend_html = '<div class="legend-grid">'
    for cls_name, hex_col in CLASS_HEX.items():
        legend_html += (
            f'<div class="legend-chip">'
            f'<span class="legend-dot" style="background:{hex_col};"></span>'
            f'{cls_name}</div>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='color:#555; font-size:0.75rem; text-align:center;'>"
        "Blood Cell Detector v1.0<br>YOLO26 · Ultralytics · Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Hero ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div style="position:relative; z-index:1;">
        <h1>🔬 Blood Cell Detector</h1>
        <p>
            An advanced <span class="accent">AI-powered diagnostic assistant</span> for real-time detection and classification 
            of cells in peripheral blood smear images. Leveraging YOLO26 architecture for precise morphology analysis.
        </p>
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">7</div>
                <div class="stat-label">Cell Classes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">87.5%</div>
                <div class="stat-label">mAP@50</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">640px</div>
                <div class="stat-label">Resolution</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">CPU</div>
                <div class="stat-label">Inference</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Image input ──────────────────────────────────────────────────────────
tab_upload, tab_samples = st.tabs(["📤  Upload Image", "🖼️  Sample Images"])

input_image: Image.Image | None = None
source_name: str = ""

with tab_upload:
    uploaded = st.file_uploader(
        "Drag & drop a blood smear image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supports JPG, PNG, BMP, TIFF. Best results with Giemsa-stained brightfield smears.",
    )
    if uploaded is not None:
        input_image = Image.open(uploaded).convert("RGB")
        source_name = uploaded.name

with tab_samples:
    sample_files = sorted(TEST_IMAGES_DIR.glob("*")) if TEST_IMAGES_DIR.exists() else []
    sample_files = [f for f in sample_files if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    if sample_files:
        cols = st.columns(min(len(sample_files), 3))
        for idx, fpath in enumerate(sample_files):
            col = cols[idx % 3]
            with col:
                thumb = Image.open(fpath).convert("RGB")
                st.image(thumb, caption=fpath.stem, use_container_width=True)
                if st.button(f"Use  {fpath.stem}", key=f"sample_{idx}", use_container_width=True):
                    input_image = thumb
                    source_name = fpath.name
    else:
        st.info("No sample images found in `test_images/` folder.")


# ── Run detection ────────────────────────────────────────────────────────
if input_image is not None:
    st.markdown("---")

    # Convert to numpy for the model
    img_np = np.array(input_image)

    with st.spinner("🔍 Running detection …"):
        model = load_model()
        results = model.predict(
            source=img_np,
            conf=confidence,
            iou=iou_threshold,
            imgsz=img_size,
            device="cpu",
            save=False,
            verbose=False,
        )
        result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    names = result.names

    # Annotate
    annotated_np = annotate_image(
        img_np, boxes, classes, confs, names,
        show_labels=show_labels,
        show_conf=show_conf,
        line_width=line_width,
    )
    annotated_pil = Image.fromarray(annotated_np)

    # ── Results layout ───────────────────────────────────────────────────
    col_img, col_stats = st.columns([3, 1])

    with col_img:
        st.markdown(f"### 📋 Detection Results — `{source_name}`")
        view_mode = st.radio(
            "View", ["Annotated", "Original", "Side by side"],
            horizontal=True, label_visibility="collapsed",
        )
        if view_mode == "Annotated":
            st.image(annotated_pil, use_container_width=True)
        elif view_mode == "Original":
            st.image(input_image, use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(input_image, caption="Original", use_container_width=True)
            with c2:
                st.image(annotated_pil, caption="Annotated", use_container_width=True)

        # Download button
        dl_bytes = pil_to_bytes(annotated_pil, "PNG")
        st.download_button(
            "⬇️  Download annotated image",
            data=dl_bytes,
            file_name=f"{Path(source_name).stem}_detected.png",
            mime="image/png",
            use_container_width=True,
        )

    with col_stats:
        summary = build_summary_table(classes, names)
        total = int(sum(summary.values()))
        max_count = max(summary.values()) if summary else 1

        st.markdown(
            f'<div class="result-card">'
            f'<h3>🧬 Detection Summary</h3>'
            f'<p style="color:#a8b2c1; font-size:0.9rem; margin-bottom:1rem;">'
            f'Total detections: <strong style="color:#e94560;">{total}</strong></p>',
            unsafe_allow_html=True,
        )

        bars_html = ""
        for cls_name, count in summary.items():
            pct = (count / max_count) * 100
            color = CLASS_HEX.get(cls_name, "#888")
            bars_html += (
                f'<div class="count-row">'
                f'<span class="count-label">{cls_name}</span>'
                f'<div class="count-bar-bg">'
                f'<div class="count-bar-fill" style="width:{pct}%; background:{color};">'
                f'{count}</div></div></div>'
            )

        st.markdown(bars_html + '</div>', unsafe_allow_html=True)

        # WBC differential
        wbc_types = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}
        wbc_counts = {k: v for k, v in summary.items() if k in wbc_types}
        wbc_total = sum(wbc_counts.values())

        if wbc_total > 0:
            st.markdown(
                '<div class="result-card"><h3>🩸 WBC Differential</h3>',
                unsafe_allow_html=True,
            )
            diff_html = ""
            for wbc_name, wbc_count in sorted(wbc_counts.items(), key=lambda x: -x[1]):
                pct = (wbc_count / wbc_total) * 100
                color = CLASS_HEX.get(wbc_name, "#888")
                diff_html += (
                    f'<div class="count-row">'
                    f'<span class="count-label">{wbc_name}</span>'
                    f'<div class="count-bar-bg">'
                    f'<div class="count-bar-fill" style="width:{pct}%; background:{color};">'
                    f'{pct:.1f}%</div></div></div>'
                )
            st.markdown(diff_html + '</div>', unsafe_allow_html=True)

    # ── Detailed table (expandable) ──────────────────────────────────────
    with st.expander("📊 Detailed detection data"):
        if len(boxes) > 0:
            import pandas as pd
            df = pd.DataFrame({
                "Class": [names[int(c)] for c in classes],
                "Confidence": [f"{c:.3f}" for c in confs],
                "x1": boxes[:, 0].astype(int),
                "y1": boxes[:, 1].astype(int),
                "x2": boxes[:, 2].astype(int),
                "y2": boxes[:, 3].astype(int),
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("No detections found. Try lowering the confidence threshold.")

else:
    # Placeholder when no image is loaded
    st.markdown(
        """
        <div style="text-align:center; padding:3rem 1rem; color:#555;">
            <p style="font-size:3rem; margin-bottom:0.5rem;">🔬</p>
            <p style="font-size:1.1rem;">Upload a blood smear image or select a sample to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:1rem; color:#555; font-size:0.8rem;">
        <strong>⚠️ Research use only</strong> — This model is not validated for clinical diagnosis.<br>
        Built with <a href="https://ultralytics.com" style="color:#e94560;">Ultralytics YOLO</a> ·
        <a href="https://streamlit.io" style="color:#e94560;">Streamlit</a> ·
        <a href="https://pytorch.org" style="color:#e94560;">PyTorch</a>
    </div>
    """,
    unsafe_allow_html=True,
)
