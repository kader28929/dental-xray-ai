import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="نظام فحص جودة صور الأشعة واستعادتها",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Cairo', sans-serif;
            direction: RTL;
            text-align: right;
        }

        .stApp {
            background: linear-gradient(180deg, #dff0fb 0%, #edf7fe 55%, #f5faff 100%);
        }

        .main .block-container {
            max-width: 1080px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-card {
            position: relative;
            background: #ffffff;
            border: 1px solid #cbe3f5;
            border-right: 6px solid #0284c7;
            border-radius: 18px;
            padding: 24px 28px;
            box-shadow: 0 8px 24px rgba(2, 132, 199, 0.08);
            margin-bottom: 22px;
        }

        .hero-header-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
        }

        .hero-content {
            flex: 1;
        }

        .uni-logo-corner {
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #f0f8ff;
            border: 1px solid #bae0fb;
            border-radius: 14px;
            padding: 8px 12px;
            min-width: 82px;
            text-align: center;
        }

        .uni-logo-corner img {
            width: 48px;
            height: 48px;
            object-fit: contain;
            margin-bottom: 4px;
        }

        .uni-logo-text {
            color: #0369a1;
            font-size: 0.72rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .hero-badge {
            display: inline-block;
            background: #e0f2fe;
            color: #0369a1;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 5px 12px;
            border-radius: 999px;
            margin-bottom: 10px;
        }

        .hero-title {
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            line-height: 1.4;
        }

        .hero-subtitle {
            color: #475569;
            font-size: 1rem;
            margin: 0;
            line-height: 1.7;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #d4e8f7;
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: 0 4px 16px rgba(2, 132, 199, 0.05);
            margin-bottom: 18px;
        }

        .section-title {
            color: #0f172a;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .diag-box {
            background: #f0f8ff;
            border: 1px solid #bae0fb;
            border-radius: 14px;
            padding: 16px 18px;
            margin-top: 12px;
        }

        .metric-pill {
            display: inline-block;
            background: #e0f2fe;
            color: #0c4a6e;
            border-radius: 10px;
            padding: 6px 12px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-left: 8px;
            margin-bottom: 8px;
        }

        [data-testid="stFileUploader"] section {
            background-color: #f3f9fe !important;
            border: 2px dashed #38bdf8 !important;
            border-radius: 16px !important;
            padding: 20px !important;
        }

        .stButton > button, .stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.65rem 1.2rem;
            font-family: 'Cairo', sans-serif;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_PATHS = [
    BASE_DIR / "best_cdae_model.h5",
    BASE_DIR / "models" / "best_cdae_model.h5",
    BASE_DIR / "models" / "models" / "best_cdae_model.h5",
]

LOGO_CANDIDATES = [
    BASE_DIR / "logo.png.WEBP",
    BASE_DIR / "logo.png",
    BASE_DIR / "al_bayan_logo.png",
    BASE_DIR / "logo.jpg",
    BASE_DIR / "logo.jpeg",
    BASE_DIR / "logo.webp",
]


def get_logo_html() -> str:
    for logo_path in LOGO_CANDIDATES:
        if logo_path.exists():
            ext = logo_path.suffix.lstrip(".").lower()
            if ext == "jpg":
                ext = "jpeg"
            elif ext == "webp":
                ext = "webp"
            b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
            return (
                f'<div class="uni-logo-corner">'
                f'<img src="data:image/{ext};base64,{b64}" alt="Al-Bayan University Logo">'
                f'<span class="uni-logo-text">جامعة البيان</span>'
                f"</div>"
            )
    return (
        '<div class="uni-logo-corner">'
        '<span style="font-size:1.5rem; line-height:1.2;">🏛️</span>'
        '<span class="uni-logo-text">جامعة البيان</span>'
        "</div>"
    )


def resolve_model_path() -> Path | None:
    for path in CANDIDATE_PATHS:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_cdae_model(model_path_str: str):
    return tf.keras.models.load_model(model_path_str, compile=False)


def is_likely_xray_image(pil_img: Image.Image) -> tuple[bool, str]:
    rgb = pil_img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    saturation = np.where(max_c > 0.05, delta / np.maximum(max_c, 1e-6), 0.0)
    high_sat_ratio = float(np.mean(saturation > 0.22))
    mean_sat = float(np.mean(saturation))

    rg_diff = float(np.mean(np.abs(r - g)))
    rb_diff = float(np.mean(np.abs(r - b)))
    gb_diff = float(np.mean(np.abs(g - b)))
    color_spread = max(rg_diff, rb_diff, gb_diff)

    if high_sat_ratio > 0.14 or mean_sat > 0.12 or color_spread > 0.045:
        return False, "هذه ليست صورة أشعة"

    gray = np.asarray(ImageOps.grayscale(rgb), dtype=np.float32) / 255.0
    if float(np.std(gray)) < 0.04:
        return False, "هذه ليست صورة أشعة"

    return True, ""


def preprocess_image(pil_img: Image.Image, target_size: tuple[int, int] = (128, 128)):
    gray = ImageOps.grayscale(pil_img)
    resized = gray.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(arr, axis=-1), axis=0)
    return arr, input_tensor


def detect_and_reconstruct_missing_regions(
    orig_arr: np.ndarray,
    pred_arr: np.ndarray,
    inpaint_strength: float = 0.65,
) -> tuple[np.ndarray, float]:
    """
    اكتشاف المناطق التالفة أو المفقودة أو المنخفضة التباين/الحدود وإعادة بنائها
    عبر دمج مخرجات CDAE وتنعيم البنية المحيطة (diffusion/inpainting) ثم رفع الدقة.
    """
    h, w = orig_arr.shape

    # اكتشاف المناطق التي فيها نقص حاد أو تشوه أو تلف أو سطوع/تباين شاذ
    local_mean_pil = Image.fromarray((orig_arr * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.BoxBlur(radius=9)
    )
    local_mean = np.asarray(local_mean_pil, dtype=np.float32) / 255.0
    local_diff = np.abs(orig_arr - local_mean)

    # مناطق داكنة جداً أو بيضاء محترقة أو ذات تباين محلي ضعيف جداً (نقص/تلف محتمل)
    dark_hole = orig_arr < 0.06
    bright_clip = orig_arr > 0.96
    low_detail = local_diff < 0.012

    defect_mask = np.clip(
        (dark_hole.astype(np.float32) * 0.85)
        + (bright_clip.astype(np.float32) * 0.75)
        + (low_detail.astype(np.float32) * 0.35),
        0.0,
        1.0,
    )

    # توسيع القناع قليلاً لضمان انتقال ناعم عند حواف المنطقة المرممة
    mask_pil = Image.fromarray((defect_mask * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.BoxBlur(radius=4)
    )
    smooth_mask = (np.asarray(mask_pil, dtype=np.float32) / 255.0) * float(np.clip(inpaint_strength, 0.0, 1.0))

    # بناء طبقة ترميم تمزج بين الصورة المستعادة بـ CDAE والامتداد المحيطي
    diffused_pil = Image.fromarray((pred_arr * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=2.0)
    )
    diffused_arr = np.asarray(diffused_pil, dtype=np.float32) / 255.0
    reconstructed_patch = np.clip(0.65 * pred_arr + 0.35 * diffused_arr, 0.0, 1.0)

    repaired_arr = np.clip((1.0 - smooth_mask) * orig_arr + smooth_mask * reconstructed_patch, 0.0, 1.0)
    defect_ratio_percent = float(np.mean(smooth_mask > 0.25) * 100.0)
    return repaired_arr, defect_ratio_percent


def restore_and_upscale_image(
    original_gray_pil: Image.Image,
    pred_2d: np.ndarray,
    blend_weight: float = 0.35,
    inpaint_strength: float = 0.65,
    upscale_factor: int = 2,
) -> tuple[Image.Image, float]:
    orig_w, orig_h = original_gray_pil.size
    orig_arr = np.asarray(original_gray_pil, dtype=np.float32) / 255.0

    pred_clipped = np.clip(pred_2d, 0.0, 1.0)
    pred_pil = Image.fromarray((pred_clipped * 255).astype(np.uint8), mode="L").resize(
        (orig_w, orig_h), Image.Resampling.BICUBIC
    )
    pred_arr = np.asarray(pred_pil, dtype=np.float32) / 255.0

    # أولاً: إعادة بناء المناطق التالفة أو الناقصة
    repaired_arr, defect_percent = detect_and_reconstruct_missing_regions(
        orig_arr=orig_arr,
        pred_arr=pred_arr,
        inpaint_strength=inpaint_strength,
    )

    # ثانياً: دمج متوازن مع مخرجات CDAE للحفاظ على تفاصيل الصورة
    w = float(np.clip(blend_weight, 0.0, 0.7))
    combined = np.clip((1.0 - w) * repaired_arr + w * pred_arr, 0.0, 1.0)

    # ثالثاً: زيادة الدقة (Resolution Upscaling) مع شحذ الحواف ومنع الضبابية
    scale = int(max(1, min(3, upscale_factor)))
    target_w = int(min(2048, orig_w * scale))
    target_h = int(min(2048, orig_h * scale))

    base_pil = Image.fromarray((combined * 255).astype(np.uint8), mode="L")
    if (target_w, target_h) != (orig_w, orig_h):
        upscaled_pil = base_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        upscaled_pil = base_pil

    # شحذ دقيق بعد رفع الدقة لإبراز حواف الأسنان والجذور بدون هالات مزعجة
    sharpened_pil = upscaled_pil.filter(ImageFilter.UnsharpMask(radius=1.35, percent=135, threshold=2))
    final_pil = ImageEnhance.Contrast(sharpened_pil).enhance(1.08)
    return final_pil, defect_percent


def estimate_image_quality(arr_2d: np.ndarray) -> dict:
    mean_val = float(np.mean(arr_2d))
    std_val = float(np.std(arr_2d))
    laplacian = np.abs(
        arr_2d[1:-1, :-2] + arr_2d[1:-1, 2:] + arr_2d[:-2, 1:-1] + arr_2d[2:, 1:-1] - 4 * arr_2d[1:-1, 1:-1]
    )
    sharpness_score = float(np.mean(laplacian))

    if mean_val < 0.22:
        issue = "نقص تعريض إشعاعي / صورة داكنة (Underexposure)"
        note = "الصورة منخفضة السطوع وقد تخفي تفاصيل المينا أو الجذور."
    elif mean_val > 0.78:
        issue = "فرط تعريض إشعاعي / صورة ساطعة جداً (Overexposure)"
        note = "زيادة السطوع قد تؤدي إلى فقدان التباين الدقيق في الأنسجة."
    elif sharpness_score < 0.035:
        issue = "تشوش أو اهتزاز حركة / ضعف حدة (Motion Blur / Defocus)"
        note = "الحواف غير حادة؛ يُنصح بمراجعة ثبات الحساس أو زمن التعريض."
    elif std_val < 0.12:
        issue = "انخفاض التباين أو ضجيج إشعاعي (Low Contrast / Noise)"
        note = "التباين محدود بين المناطق التشريحية المختلفة."
    else:
        issue = "جودة أولية مقبولة مع فحص تحسيني وقائي"
        note = "الصورة متوازنة نسبياً، وتم تطبيق إعادة بناء وتحسين دقة متوازن."

    quality_percent = int(max(15, min(98, round((sharpness_score * 900) + (std_val * 140)))))
    return {
        "issue": issue,
        "note": note,
        "quality_percent": quality_percent,
        "brightness_percent": int(round(mean_val * 100)),
        "contrast_percent": int(round(min(100.0, std_val * 220))),
    }


logo_block = get_logo_html()

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-header-row">
            <div class="hero-content">
                <div class="hero-badge">مشروع تخرج — هندسة تقنيات الأجهزة الطبية</div>
                <h1 class="hero-title">🦷 نظام فحص جودة صور الأشعة واستعادتها بالذكاء الاصطناعي</h1>
                <p class="hero-subtitle">
                    منصة بحثية لتحليل صور الأشعة السينية داخل الفموية (Intraoral X-ray)، والتحقق من نوع الصورة، وإعادة بناء النقص أو الأجزاء التالفة، ورفع الدقة مع الحفاظ على وضوح التفاصيل.
                </p>
            </div>
            {logo_block}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model_path = resolve_model_path()

if model_path is None:
    st.error("تعذر العثور على ملف النموذج `best_cdae_model.h5` داخل المستودع.")
    st.stop()

try:
    model = load_cdae_model(str(model_path))
except Exception as exc:
    st.error(f"حدث خطأ أثناء تحميل النموذج: {exc}")
    st.stop()

input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]
target_h = int(input_shape[1]) if len(input_shape) >= 3 and input_shape[1] else 128
target_w = int(input_shape[2]) if len(input_shape) >= 3 and input_shape[2] else 128

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📤 رفع صورة الأشعة السينية وإعدادات الاستعادة والدقة</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "اختر صورة أشعة سينية (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
with c_ctrl1:
    blend_weight = st.slider(
        "نسبة تأثير نموذج CDAE",
        min_value=0.10,
        max_value=0.65,
        value=0.30,
        step=0.05,
    )
with c_ctrl2:
    inpaint_strength = st.slider(
        "قوة إعادة بناء النقص والتلف",
        min_value=0.10,
        max_value=1.00,
        value=0.65,
        step=0.05,
    )
with c_ctrl3:
    upscale_factor = st.selectbox(
        "معامل زيادة الدقة (Resolution)",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: f"{x}x (الدقة الأصلية)" if x == 1 else f"{x}x (رفع الدقة)",
    )

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    original_pil = Image.open(uploaded_file)

    is_xray, reject_msg = is_likely_xray_image(original_pil)
    if not is_xray:
        st.error(f"❌ {reject_msg}")
        st.warning("يرجى رفع صورة أشعة سينية خاصة بالأسنان (رمادية/إشعاعية) فقط.")
        st.stop()

    orig_gray_pil = ImageOps.grayscale(original_pil)
    arr_2d, input_tensor = preprocess_image(original_pil, target_size=(target_w, target_h))
    metrics = estimate_image_quality(arr_2d)

    with st.spinner("جاري إعادة بناء النقص ورفع دقة الصورة..."):
        pred = model.predict(input_tensor, verbose=0)
        if isinstance(pred, list):
            pred = pred[0]
        pred_2d = np.squeeze(pred)
        restored_pil, defect_percent = restore_and_upscale_image(
            original_gray_pil=orig_gray_pil,
            pred_2d=pred_2d,
            blend_weight=blend_weight,
            inpaint_strength=inpaint_strength,
            upscale_factor=upscale_factor,
        )

    orig_w, orig_h = orig_gray_pil.size
    res_w, res_h = restored_pil.size

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">🩻 الصورة الأصلية ({orig_w}×{orig_h} px)</div>',
            unsafe_allow_html=True,
        )
        st.image(original_pil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">✨ الصورة المستعادة وعالية الدقة ({res_w}×{res_h} px)</div>',
            unsafe_allow_html=True,
        )
        st.image(restored_pil, use_container_width=True)

        buf = BytesIO()
        restored_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ تحميل الصورة المستعادة وعالية الدقة (PNG)",
            data=buf.getvalue(),
            file_name="reconstructed_highres_dental_xray.png",
            mime="image/png",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 تقرير الفحص وإعادة البناء والدقة</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <span class="metric-pill">مؤشر الوضوح التقريبي: {metrics["quality_percent"]}%</span>
        <span class="metric-pill">نسبة المناطق المرممة: {defect_percent:.1f}%</span>
        <span class="metric-pill">السطوع: {metrics["brightness_percent"]}%</span>
        <span class="metric-pill">التباين: {metrics["contrast_percent"]}%</span>
        <div class="diag-box">
            <strong>الحالة المرصودة:</strong> {metrics["issue"]}<br>
            <span style="color:#475569;">{metrics["note"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("ملاحظة بحثية: هذا النظام مخصص للأغراض الأكاديمية والبحثية لمشروع التخرج ولا يُغني عن التقييم السريري المتخصص.")
    st.markdown("</div>", unsafe_allow_html=True)
