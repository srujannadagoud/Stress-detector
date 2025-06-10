import streamlit as st
import cv2
import numpy as np
from PIL import Image

def analyze_part(image):
    image_np = np.array(image.convert('RGB'))
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image_cv.copy()
    suggestions = []

    # Find main part contour
    max_area = 0
    main_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            main_contour = contour

    dimensions = None
    if main_contour is not None:
        x, y, w, h = cv2.boundingRect(main_contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dimensions = (w, h)

    # Detect circular holes
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30, minRadius=5, maxRadius=60)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = i
            cv2.circle(output, (cx, cy), r, (255, 0, 0), 2)
            suggestions.append(f"🟢 Circular hole at ({cx}, {cy}), radius: {r}px")

    # Detect rectangular or slot-like features
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.drawContours(output, [approx], -1, (0, 255, 255), 2)
            suggestions.append(f"🟡 Possible slot at ({x}, {y}) - size: {w}px x {h}px")

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output, suggestions, dimensions

# --- Streamlit App ---
st.set_page_config(page_title="Stress Detector", layout="wide")
st.title("🛠 Simple Stress Feature Detector")

uploaded_file = st.file_uploader("📂 Upload an image of a mechanical part", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("🔍 Processing..."):
        result_img, tips, dims = analyze_part(image)

    st.image(result_img, caption="📌 Detected Features", use_column_width=True)

    if dims:
        st.markdown("### 📏 Estimated Part Dimensions (pixels)")
        st.write(f"- Width: **{dims[0]} px**")
        st.write(f"- Height: **{dims[1]} px**")

    st.markdown("### 💡 Suggestions:")
    if tips:
        for tip in tips:
            st.write(tip)
    else:
        st.success("✅ No prominent stress concentrators found.")