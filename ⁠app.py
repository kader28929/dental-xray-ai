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
            background: linear-gradient(180deg, #f4f8fb 0%, #ffffff 100%);
        }

        .main .block-container {
            max-width: 1050px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-card {
            background: #ffffff;
            border: 1px solid #e2ecf3;
            border-right: 6px solid #0ea5e9;
            border-radius: 18px;
            padding: 24px 28px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 22px;
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
            font-size: 1.85rem;
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
            border: 1px solid #e6eef5;
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
            margin-bottom: 18px;
        }

        .section-title {
            color: #0f172a;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .diag-box {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            padding: 16px 18px;
            margin-top: 12px;
        }

        .metric-pill {
            display: inline-block;
            background: #f1f5f9;
            color: #1e293b;
            border-radius: 10px;
            padding: 6px 12px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-left: 8px;
            margin-bottom: 8px;
        }

        [data-testid="stFileUploader"] section {
            background-color: #f8fbff !important;
            border: 2px dashed #7dd3fc !important;
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


def resolve_model_path() -> Path | None:
    for path in CANDIDATE_PATHS:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_cdae_model(model_path_str: str):
    return tf.keras.models.load_model(model_path_str, compile=False)


def is_likely_xray_image(pil_img: Image.Image) -> tuple[bool, str]:
    """
    فحص أولي للتحقق أن الصورة أقرب إلى صورة أشعة سينية وليست صورة ملونة طبيعية أو غير إشعاعية.
    """
    rgb = pil_img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # حساب نسبة التشبع اللوني الحقيقي
    saturation = np.where(max_c > 0.05, delta / np.maximum(max_c, 1e-6), 0.0)
    high_sat_ratio = float(np.mean(saturation > 0.22))
    mean_sat = float(np.mean(saturation))

    # فحص الفارق الكبير بين القنوات اللونية (الصور الطبيعية والواجهات الملونة)
    rg_diff = float(np.mean(np.abs(r - g)))
    rb_diff = float(np.mean(np.abs(r - b)))
    gb_diff = float(np.mean(np.abs(g - b)))
    color_spread = max(rg_diff, rb_diff, gb_diff)

    # رفض الصور ذات التشبع اللوني الواضح أو التباين اللوني العالي
    if high_sat_ratio > 0.14 or mean_sat > 0.12 or color_spread > 0.045:
        return False, "هذه ليست صورة أشعة"

    # التأكد من أن الصورة ليست مسطحة بالكامل أو ضجيجاً عشوائياً فارغاً
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


def enhance_without_blur(original_gray_pil: Image.Image, pred_2d: np.ndarray, blend_weight: float = 0.35) -> Image.Image:
    """
    دمج مخرجات CDAE مع الصورة الأصلية بدقتها الكاملة وشحذ التفاصيل لتجنب الضبابية.
    """
    orig_w, orig_h = original_gray_pil.size
    orig_arr = np.asarray(original_gray_pil, dtype=np.float32) / 255.0

    pred_clipped = np.clip(pred_2d, 0.0, 1.0)
    pred_pil = Image.fromarray((pred_clipped * 255).astype(np.uint8), mode="L").resize(
        (orig_w, orig_h), Image.Resampling.BICUBIC
    )
    pred_arr = np.asarray(pred_pil, dtype=np.float32) / 255.0

    # الحفاظ على وضوح الصورة الأصلية وإضافة تحسين النموذج بنسب متوازنة
    w = float(np.clip(blend_weight, 0.0, 0.7))
    combined = np.clip((1.0 - w) * orig_arr + w * pred_arr, 0.0, 1.0)

    out_pil = Image.fromarray((combined * 255).astype(np.uint8), mode="L")
    # شحذ الحواف الدقيقة لإبراز التفاصيل التشريحية
    out_pil = out_pil.filter(ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=2))
    out_pil = ImageEnhance.Contrast(out_pil).enhance(1.08)
    return out_pil


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
        note = "الصورة متوازنة نسبياً، وتم تطبيق تحسين متوازن يحافظ على حدة التفاصيل."

    quality_percent = int(max(15, min(98, round((sharpness_score * 900) + (std_val * 140)))))
    return {
        "issue": issue,
        "note": note,
        "quality_percent": quality_percent,
        "brightness_percent": int(round(mean_val * 100)),
        "contrast_percent": int(round(min(100.0, std_val * 220))),
    }


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">مشروع تخرج — هندسة تقنيات الأجهزة الطبية</div>
        <h1 class="hero-title">🦷 نظام فحص جودة صور الأشعة واستعادتها بالذكاء الاصطناعي</h1>
        <p class="hero-subtitle">
            منصة بحثية لتحليل صور الأشعة السينية داخل الفموية (Intraoral X-ray)، والتحقق من نوع الصورة، وتشخيص المؤشرات الأولية للخلل أو ضعف الجودة مع الحفاظ على وضوح التفاصيل.
        </p>
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
st.markdown('<div class="section-title">📤 رفع صورة الأشعة السينية للفحص والاستعادة</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "اختر صورة أشعة سينية (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

blend_weight = st.slider(
    "نسبة تأثير نموذج الاستعادة (أقل = أوضح وأحدّ، أعلى = معالجة أقوى)",
    min_value=0.10,
    max_value=0.65,
    value=0.30,
    step=0.05,
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

    with st.spinner("جاري فحص الصورة واستعادتها بدون فقدان الوضوح..."):
        pred = model.predict(input_tensor, verbose=0)
        if isinstance(pred, list):
            pred = pred[0]
        pred_2d = np.squeeze(pred)
        restored_pil = enhance_without_blur(orig_gray_pil, pred_2d, blend_weight=blend_weight)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🩻 الصورة الأصلية المرفوعة</div>', unsafe_allow_html=True)
        st.image(original_pil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">✨ الصورة بعد المعالجة والتحسين</div>', unsafe_allow_html=True)
        st.image(restored_pil, use_container_width=True)

        buf = BytesIO()
        restored_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ تحميل الصورة المحسّنة (PNG)",
            data=buf.getvalue(),
            file_name="enhanced_dental_xray.png",
            mime="image/png",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 تقرير الفحص والتشخيص الأولي</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <span class="metric-pill">مؤشر الوضوح التقريبي: {metrics["quality_percent"]}%</span>
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
