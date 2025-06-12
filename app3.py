import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Material Detection Function ---
def detect_material(image_np):
    avg_color = np.mean(image_np.reshape(-1, 3), axis=0)
    r, g, b = avg_color
    hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv

    if 20 < h < 45 and s > 60 and v > 100:
        return "🟡 Brass or Gold-Plated Metal"
    elif r > 200 and g > 200 and b > 200 and max(r, g, b) - min(r, g, b) < 20:
        return "⚪ Polished Aluminum or Chrome"
    elif r < 90 and g < 90 and b < 90:
        return "⚫ Mild Steel, Cast Iron, or Carbon Alloy"
    elif s < 40 and 100 < v < 200:
        return "🔘 Oxidized or Matte-Coated Metal"
    else:
        return "⚠️ Unknown Material Type (Composite or CAD-like)"

# --- Blur Detection ---
def is_blurry(image_np, threshold=100):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, round(fm, 2)

# --- Stress Concentration Detection ---
def detect_stress_concentration(image_np):
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = image_cv.copy()
    tips = []

    holes_detected = 0
    stress_like_regions = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Detect circular holes (geometry stress points)
        if len(approx) > 8 and 100 < area < 1000:
            cv2.circle(output, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)
            holes_detected += 1

        # Detect large regions that may be stress zones
        elif area > 150 and w > 10 and h > 10:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            stress_like_regions.append((x, y, w, h))

    if not stress_like_regions and holes_detected >= 2:
        reason = "ℹ️ No visible stress detected, but hole patterns or corners may act as stress risers under load."
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), [], reason

    if not stress_like_regions and holes_detected == 0:
        reason = "⚠️ No stress or geometric features detected. Image may be CAD model or too clean."
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), [], reason

    tips = [f"🔴 Stress-like region at ({x}, {y}) — approx {w}×{h}px" for (x, y, w, h) in stress_like_regions]
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), tips, None

# --- Streamlit UI ---
st.set_page_config(page_title="AI Material & Stress Detection", layout="wide")
st.title("🧠 AI-Based Material & Stress Concentration Detector")
st.markdown("Upload an image of a mechanical part (CAD, real photo, or simulation) to detect **material type**, **stress concentration**, and **image quality**.")

uploaded_file = st.file_uploader("📁 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing..."):
        material = detect_material(image_np)
        blurry, sharpness_score = is_blurry(image_np)
        stress_img, tips, reason = detect_stress_concentration(image_np)

    st.markdown(f"### 🧾 Detected Material: {material}")

    if blurry:
        st.error(f"📛 Image is blurry (Sharpness score: {sharpness_score}). Try uploading a clearer image for better analysis.")
    else:
        st.success(f"✅ Image sharpness is good (Score: {sharpness_score})")

    st.image(stress_img, caption="📌 Stress Zones / Geometry Highlights", use_column_width=True)

    st.markdown("### 📌 Stress Analysis Output:")
    if tips:
        for tip in tips:
            st.write(tip)
    else:
        st.info(reason or "No stress zones detected.")