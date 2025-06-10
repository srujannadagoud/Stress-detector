import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Enhanced Material Detection with More Classes ---
def detect_material(image_np):
    avg_color = np.mean(image_np.reshape(-1, 3), axis=0)
    r, g, b = avg_color
    hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv

    if 20 < h < 45 and s > 60 and v > 100:
        return "🟡 Brass or Gold-Plated Metal"
    elif r > 200 and g > 200 and b > 200 and max(r, g, b) - min(r, g, b) < 20:
        return "⚪ Polished Aluminum or Chrome"
    elif 90 < h < 160 and s > 40:
        return "🟢 Painted Surface (Green Pigment)"
    elif 200 < h < 250 and s > 40:
        return "🔵 Painted Surface (Blue Pigment)"
    elif 0 < h < 15 and s > 40:
        return "🔴 Painted Surface (Red Pigment)"
    elif r < 90 and g < 90 and b < 90:
        return "⚫ Cast Iron, Steel, or Carbon Material"
    elif s < 40 and 100 < v < 200:
        return "🔘 Coated or Oxidized Metal Surface"
    else:
        return "⚠️ Unknown or Composite Material - Appearance not in database"

# --- Stress Detection Function ---
def detect_stress_concentration(image_np):
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image_cv.copy()
    tips = []

    filtered = [cnt for cnt in contours if 150 < cv2.contourArea(cnt) < 10000]
    sorted_cnts = sorted(filtered, key=cv2.contourArea, reverse=True)[:5]

    if not sorted_cnts:
        reason = "⚠️ No significant stress concentration regions found. Possible reasons:\n- Image lacks visible deformation or stress marks.\n- Resolution is too low or background dominates.\n- Image might be purely CAD/3D without stress fields."
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), [], reason

    for cnt in sorted_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        tips.append(f"🔴 High-stress zone at ({x}, {y}) — approx size: {w}px × {h}px")

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), tips, None

# --- Streamlit App ---
st.set_page_config(page_title="Material & Stress Detector", layout="wide")
st.title("🛠️ AI-Powered Material and Stress Concentration Detector")

uploaded_file = st.file_uploader("📂 Upload an image of a mechanical part (2D or 3D)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting material and stress concentration..."):
        material = detect_material(image_np)
        stress_img, tips, reason = detect_stress_concentration(image_np)

    st.markdown(f"### 🧾 **Detected Material:** {material}")
    st.image(stress_img, caption="📌 Stress Concentration Areas", use_column_width=True)

    st.markdown("### 💡 **Stress Analysis Result:**")
    if tips:
        for tip in tips:
            st.write(tip)
    else:
        st.warning(reason or "No stress zones detected.")
