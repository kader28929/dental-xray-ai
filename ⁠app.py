from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Dental X-ray AI Restoration & Quality Check",
    layout="wide",
)

st.title("🦷 نظام فحص جودة صور الأشعة واستعادتها بالذكاء الاصطناعي")
st.write("نموذج بحثي لمشروع التخرج — قسم هندسة تقنيات الأجهزة الطبية")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_cdae_model.h5"


@st.cache_resource
def load_ai_model(model_path: Path):
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(str(model_path), compile=False)


model = load_ai_model(MODEL_PATH)

if model is None:
    st.error(f"ملف النموذج غير موجود حالياً في المسار: {MODEL_PATH}")
    st.info(
        "ملاحظة: يجب وضع ملف best_cdae_model.h5 داخل مجلد اسمه models بجانب ملف app.py."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "اختر صورة أشعة سينية (PNG أو JPG أو JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (256, 256))
    img_norm = img_resized.astype(np.float32) / 255.0

    variance = cv2.Laplacian(img_resized, cv2.CV_64F).var()
    contrast = img_resized.std()

    st.markdown("---")
    st.subheader("📋 تقرير الفحص والمعالجة")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### 🔍 مؤشرات جودة الصورة")
        if contrast < 30:
            st.warning(
                "⚠️ انخفاض التباين (Low Contrast): قد يرتبط بإعدادات التعرض أو التباين."
            )
        else:
            st.success("✅ مستوى التباين ضمن النطاق المقبول.")

        if variance < 80:
            st.warning(
                "⚠️ انخفاض الحدة / ضبابية (Blur / Noise): قد يرتبط بضبابية أو تشويش في الصورة."
            )
        else:
            st.success("✅ حدة التفاصيل ضمن النطاق المقبول.")

    with col_r2:
        st.markdown("#### 🛠️ ملاحظة منهجية")
        st.info(
            "هذا تحليل أولي لجودة الصورة؛ ولا يُعد تأكيداً مستقلاً لوجود عطل فيزيائي في جهاز الأشعة أو تشخيصاً سريرياً."
        )

    restored_tensor = model.predict(img_norm.reshape(1, 256, 256, 1), verbose=0)
    restored_img = np.clip(restored_tensor.squeeze() * 255.0, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    restored_img = clahe.apply(restored_img)

    gamma = 1.35
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype="uint8",
    )
    restored_img = cv2.LUT(restored_img, table)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    restored_img = cv2.filter2D(restored_img, -1, kernel)

    st.markdown("---")
    st.subheader("🖼️ المقارنة البصرية (الأصلية مقابل المعالجة)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, caption="الصورة الأصلية", use_container_width=True)
    with col2:
        st.image(
            restored_img,
            caption="الصورة بعد الترميم والتحسين",
            use_container_width=True,
        )

    _, encoded_png = cv2.imencode(".png", restored_img)
    st.download_button(
        label="📥 تحميل الصورة المعالجة",
        data=encoded_png.tobytes(),
        file_name="repaired_dental_xray.png",
        mime="image/png",
    )
