import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceIQ · Gender & Age Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .stApp { background: #0a0a0f; color: #e8e8e8; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px; }

  .hero { text-align: center; padding: 2.5rem 0 1.5rem; }
  .hero-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #00ffaa;
    background: rgba(0,255,170,0.08);
    border: 1px solid rgba(0,255,170,0.25);
    padding: 4px 14px;
    border-radius: 2px;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
  }
  .hero h1 {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0 0 0.5rem;
  }
  .hero h1 span { color: #00ffaa; }
  .hero p { color: #777; font-size: 1rem; max-width: 500px; margin: 0 auto; }

  .tech-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem; justify-content: center; }
  .badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #777;
    padding: 3px 10px;
    border-radius: 3px;
  }
  .sep { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.5rem 0; }

  .face-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.7rem 0;
    position: relative;
    overflow: hidden;
  }
  .face-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #5ba3ff;
  }
  .face-card.female::before { background: #ff6eb4; }

  .face-num { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #555; letter-spacing: 0.15em; margin-bottom: 0.4rem; }
  .face-gender { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
  .male-txt   { color: #5ba3ff; }
  .female-txt { color: #ff6eb4; }
  .face-age   { font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #aaa; margin-top: 0.3rem; }

  .conf-bar-wrap { margin-top: 0.7rem; background: rgba(255,255,255,0.05); border-radius: 999px; height: 4px; width: 100%; }
  .conf-bar  { height: 4px; border-radius: 999px; background: #00ffaa; }
  .conf-label { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: #555; margin-top: 4px; }

  .stats-row { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
  .stat-pill {
    flex: 1; min-width: 100px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
  }
  .stat-pill .val { font-size: 1.8rem; font-weight: 800; color: #00ffaa; }
  .stat-pill .lbl { font-family: 'Space Mono', monospace; font-size: 0.58rem; letter-spacing: 0.15em; color: #555; text-transform: uppercase; }

  .stButton > button {
    background: #00ffaa !important; color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
    font-weight: 700 !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; border: none !important;
    padding: 0.7rem 2rem !important; border-radius: 4px !important; width: 100% !important;
  }
  .stButton > button:hover { background: #00e699 !important; box-shadow: 0 4px 20px rgba(0,255,170,0.3) !important; }
  .stSpinner > div { border-top-color: #00ffaa !important; }
  div[data-testid="stFileUploader"] label { display: none; }

  .warning-box {
    background: rgba(255,180,0,0.08);
    border: 1px solid rgba(255,180,0,0.25);
    border-radius: 8px; padding: 1rem 1.2rem;
    font-size: 0.85rem; color: #ccc; margin: 1rem 0;
  }
  .warning-box code {
    background: rgba(255,255,255,0.1); padding: 2px 6px;
    border-radius: 3px; font-family: 'Space Mono', monospace;
    font-size: 0.8rem; color: #00ffaa;
  }
</style>
""", unsafe_allow_html=True)

# ── Model paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_PROTO   = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL   = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO    = os.path.join(BASE_DIR, "age_deploy.prototxt")
AGE_MODEL    = os.path.join(BASE_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(BASE_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(BASE_DIR, "gender_net.caffemodel")

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN  = (78.4263377603, 87.7689143744, 114.895847746)

# ── Helpers ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    face_net   = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net    = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net


def models_present():
    return all(os.path.exists(p) for p in [
        FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL
    ])


def detect_faces(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN, swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2, y2])
    return boxes


def predict_age_gender(age_net, gender_net, face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN, swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_idx   = gender_preds[0].argmax()
    gender       = GENDER_LIST[gender_idx]
    gender_conf  = float(gender_preds[0][gender_idx])

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_idx   = age_preds[0].argmax()
    age       = AGE_BUCKETS[age_idx]
    age_conf  = float(age_preds[0][age_idx])

    return gender, gender_conf, age, age_conf


def annotate_image(frame, boxes, results):
    out = frame.copy()
    for box, res in zip(boxes, results):
        x1, y1, x2, y2 = box
        gender, g_conf, age, _ = res
        color = (91, 163, 255) if gender == 'Male' else (255, 110, 180)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        cl = max(10, (x2 - x1) // 6)
        for (px, py) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = cl if px == x1 else -cl
            dy = cl if py == y1 else -cl
            cv2.line(out, (px, py), (px + dx, py), color, 3)
            cv2.line(out, (px, py), (px, py + dy), color, 3)

        label = f"{gender}  |  {age} yrs"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        ly = max(y1 - lh - 12, 0)
        cv2.rectangle(out, (x1, ly), (x1 + lw + 14, ly + lh + 12), color, -1)
        cv2.putText(out, label, (x1 + 7, ly + lh + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (15, 15, 25), 1, cv2.LINE_AA)
    return out


# ── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">Deep Learning Mini Project</div>
  <h1>Face<span>IQ</span></h1>
  <p>Upload a photo — AI detects every face and predicts gender & age using pre-trained Caffe deep learning models.</p>
  <div class="tech-badges">
    <span class="badge">OpenCV DNN</span>
    <span class="badge">Caffe Models</span>
    <span class="badge">Adience Dataset</span>
    <span class="badge">Streamlit UI</span>
    <span class="badge">Python 3</span>
  </div>
</div>
<hr class="sep">
""", unsafe_allow_html=True)

# ── Model missing warning ─────────────────────────────────────────────────────────
if not models_present():
    missing = [os.path.basename(p) for p in [
        FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL
    ] if not os.path.exists(p)]
    st.markdown(f"""
<div class="warning-box">
  ⚠️ <strong>Model files missing.</strong> Clone the reference repo and copy these files next to <code>app.py</code>:<br><br>
  {'  '.join([f'<code>{f}</code>' for f in missing])}<br><br>
  <code>git clone https://github.com/smahesh29/Gender-and-Age-Detection</code><br>
  Then copy all <code>.prototxt</code>, <code>.caffemodel</code>, <code>.pb</code> and <code>.pbtxt</code> files alongside this script.
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("#### 📤 Upload Image")
    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        if max(pil_img.size) > 1024:
            ratio   = 1024 / max(pil_img.size)
            pil_img = pil_img.resize((int(pil_img.width*ratio), int(pil_img.height*ratio)), Image.LANCZOS)

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)
        run_btn = st.button("🔍 Detect & Analyze Faces", disabled=not models_present())
    else:
        st.markdown("""
<div style="border:2px dashed rgba(0,255,170,0.2);border-radius:12px;padding:2.5rem;
     text-align:center;background:rgba(0,255,170,0.02);margin-top:0.5rem;">
  <div style="font-size:2.5rem;margin-bottom:0.5rem;">📸</div>
  <div style="color:#555;font-size:0.9rem;">Drag & drop or click to upload</div>
  <div style="color:#3a3a3a;font-size:0.75rem;margin-top:0.3rem;">JPG · PNG · WEBP</div>
</div>""", unsafe_allow_html=True)
        run_btn = False

with col_right:
    st.markdown("#### 📊 Results")

    if not uploaded:
        st.markdown('<div style="color:#333;padding:3rem 0;text-align:center;font-size:0.9rem;">Upload an image to see results.</div>',
                    unsafe_allow_html=True)
    elif run_btn:
        with st.spinner("Loading models…"):
            face_net, age_net, gender_net = load_models()

        with st.spinner("Detecting faces…"):
            boxes = detect_faces(face_net, img_bgr)

        if not boxes:
            st.warning("No faces detected. Try a clearer front-facing photo.")
        else:
            results = []
            h, w    = img_bgr.shape[:2]
            padding = 20

            with st.spinner(f"Predicting gender & age for {len(boxes)} face(s)…"):
                for (x1, y1, x2, y2) in boxes:
                    fx1 = max(0, x1 - padding)
                    fy1 = max(0, y1 - padding)
                    fx2 = min(w, x2 + padding)
                    fy2 = min(h, y2 + padding)
                    face_crop = img_bgr[fy1:fy2, fx1:fx2]
                    results.append(predict_age_gender(age_net, gender_net, face_crop))

            males   = sum(1 for r in results if r[0] == 'Male')
            females = len(results) - males

            st.markdown(f"""
<div class="stats-row">
  <div class="stat-pill"><div class="val">{len(results)}</div><div class="lbl">Faces</div></div>
  <div class="stat-pill"><div class="val" style="color:#5ba3ff">{males}</div><div class="lbl">Male</div></div>
  <div class="stat-pill"><div class="val" style="color:#ff6eb4">{females}</div><div class="lbl">Female</div></div>
</div>""", unsafe_allow_html=True)

            for i, (gender, g_conf, age, a_conf) in enumerate(results):
                card_cls   = "female" if gender == "Female" else ""
                gender_cls = "female-txt" if gender == "Female" else "male-txt"
                icon       = "♂" if gender == "Male" else "♀"
                bar_w      = int(g_conf * 100)
                st.markdown(f"""
<div class="face-card {card_cls}">
  <div class="face-num">FACE #{i+1}</div>
  <div class="face-gender {gender_cls}">{icon} {gender}</div>
  <div class="face-age">Age: {age} years</div>
  <div class="conf-bar-wrap"><div class="conf-bar" style="width:{bar_w}%"></div></div>
  <div class="conf-label">Gender Confidence: {g_conf*100:.1f}%</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<hr class='sep'>", unsafe_allow_html=True)
            st.markdown("#### 🖼 Annotated Output")
            annotated_bgr = annotate_image(img_bgr, boxes, results)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

            buf = io.BytesIO()
            Image.fromarray(annotated_rgb).save(buf, format="PNG")
            st.download_button("⬇ Download Annotated Image", data=buf.getvalue(),
                               file_name="faceiq_result.png", mime="image/png")

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
**Step 1 — Face Detection**  
`opencv_face_detector_uint8.pb` — a TensorFlow SSD model finds all face bounding boxes.

**Step 2 — Gender Prediction**  
Each face crop → `gender_net.caffemodel` (CNN trained on Adience) → Male / Female + confidence score.

**Step 3 — Age Prediction**  
Same crop → `age_net.caffemodel` — classifies into one of 8 age buckets:  
(0–2), (4–6), (8–12), (15–20), (25–32), (38–43), (48–53), (60–100)

**Models:** Tal Hassner & Gil Levi  
**Dataset:** Adience (26,580 face images)
    """)
    st.markdown("---")
    st.markdown("**Quick Setup**")
    st.code("""# 1. Clone the model repo
git clone https://github.com/smahesh29/\\
  Gender-and-Age-Detection

# 2. Copy model files next to app.py
cp Gender-and-Age-Detection/*.pb .
cp Gender-and-Age-Detection/*.pbtxt .
cp Gender-and-Age-Detection/*.prototxt .
cp Gender-and-Age-Detection/*.caffemodel .

# 3. Install & run
pip install streamlit opencv-python numpy Pillow
streamlit run app.py""", language="bash")
    st.markdown("---")
    st.caption("Deep Learning Mini Project · 2025")
