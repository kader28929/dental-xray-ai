import os
import io
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Dental Intraoral X-Ray Restoration - Al-Bayan University",
    page_icon="🦷",
    layout="wide",
)

MODEL_PATH = "best_cdae_model.h5"
LOGO_CANDIDATES = ["logo.png.WEBP", "bayan_logo.png"]

# تنسيق الواجهة باللون الأزرق الفاتح والهوية البصرية
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #e8f4fc 0%, #f5fafe 55%, #ffffff 100%);
        }
        [data-testid="stSidebar"] {
            background-color: #dceefa;
            border-right: 1px solid #b8dcf5;
        }
        .main-header {
            background-color: #d6ecfa;
            border: 1px solid #a8d4f0;
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 18px;
        }
        .main-header h1 {
            color: #0b3c5d;
            margin: 0 0 6px 0;
            font-size: 1.75rem;
        }
        .main-header p {
            color: #245678;
            margin: 0;
            font-size: 0.98rem;
        }
        .stButton > button, .stDownloadButton > button {
            background-color: #1d70b8;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #14538a;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_logo_path() -> str | None:
    for candidate in LOGO_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource(show_spinner="جاري تحميل نموذج الذكاء الاصطناعي...")
def load_cdae_model(path: str):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)


def enhance_radiograph_and_black_level(
    pil_gray: Image.Image,
    black_boost: float = 0.12,
    apply_clarity: bool = True,
) -> Image.Image:
    arr = np.asarray(pil_gray, dtype=np.float32) / 255.0

    if black_boost > 0.0:
        low_thresh = np.clip(black_boost, 0.0, 0.45)
        arr = np.clip((arr - low_thresh) / (1.0 - low_thresh), 0.0, 1.0)
        arr = np.power(arr, 1.15)

    out_img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")

    if apply_clarity:
        out_img = ImageOps.autocontrast(out_img, cutoff=1)
        out_img = out_img.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=90, threshold=2)
        )

    return out_img


def restore_dental_xray(
    uploaded_pil: Image.Image,
    model,
    scale_factor: int = 1,
    black_boost: float = 0.12,
    apply_clarity: bool = True,
) -> Image.Image:
    orig_w, orig_h = uploaded_pil.size
    gray_256 = ImageOps.grayscale(uploaded_pil).resize(
        (256, 256), Image.Resampling.BICUBIC
    )
    inp_arr = np.asarray(gray_256, dtype=np.float32) / 255.0
    inp_tensor = inp_arr[np.newaxis, ..., np.newaxis]

    pred_arr = model.predict(inp_tensor, verbose=0)[0, ..., 0]
    pred_uint8 = np.clip(pred_arr * 255.0, 0, 255).astype(np.uint8)
    restored_256 = Image.fromarray(pred_uint8, mode="L")

    enhanced_256 = enhance_radiograph_and_black_level(
        restored_256,
        black_boost=black_boost,
        apply_clarity=apply_clarity,
    )

    target_w = max(256, orig_w) * int(scale_factor)
    target_h = max(256, orig_h) * int(scale_factor)
    return enhanced_256.resize((target_w, target_h), Image.Resampling.LANCZOS)


logo_file = find_logo_path()

with st.sidebar:
    if logo_file is not None:
        st.image(logo_file, use_container_width=True)
    st.markdown("### 🦷 ترميم صور الأشعة السنية")
    st.caption("مشروع تخرج - كلية طب الأسنان / جامعة البيان")
    st.divider()

    scale_factor = st.selectbox("معامل التكبير (Resolution)", options=[1, 2, 3], index=0)
    apply_clarity = st.checkbox("تحسين التباين وإبراز التفاصيل", value=True)
    black_boost = st.slider(
        "تعزيز اللون الأسود (Black Level Boost)",
        min_value=0.0,
        max_value=0.35,
        value=0.12,
        step=0.02,
        help="يزيد عمق اللون الأسود ويزيل الغشاوة الرمادية الخفيفة حول الصورة وبين الجذور.",
    )

st.markdown(
    """
    <div class="main-header">
        <h1>🦷 نظام استعادة وإعادة بناء صور الأشعة السنية (CDAE)</h1>
        <p>ترميم الصور التالفة أو الناقصة، إزالة التشويش، وتعزيز التباين واللون الأسود بدقة 256×256</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_cdae_model(MODEL_PATH)
if model is None:
    st.error(
        f"لم يتم العثور على ملف النموذج `{MODEL_PATH}` في المستودع. تأكد من رفعه إلى الجذر الرئيسي على GitHub."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "اختر صورة الأشعة (PNG, JPG, JPEG, WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is not None:
    original_img = Image.open(uploaded_file)

    with st.spinner("جاري إعادة بناء وترميم الصورة وتعزيز التباين..."):
        restored_img = restore_dental_xray(
            uploaded_pil=original_img,
            model=model,
            scale_factor=scale_factor,
            black_boost=black_boost,
            apply_clarity=apply_clarity,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("الصورة الأصلية")
        st.image(original_img, use_container_width=True)
        st.caption(f"الأبعاد الأصلية: {original_img.size[0]} × {original_img.size[1]} px")

    with col2:
        st.subheader("الصورة المُرمَّمة والمعزّزة")
        st.image(restored_img, use_container_width=True)
        st.caption(f"الأبعاد الناتجة: {restored_img.size[0]} × {restored_img.size[1]} px")

        img_buffer = io.BytesIO()
        restored_img.save(img_buffer, format="PNG")
        st.download_button(
            label="⬇️ تحميل الصورة المُرمَّمة (PNG)",
            data=img_buffer.getvalue(),
            file_name="restored_dental_xray.png",
            mime="image/png",
        )
