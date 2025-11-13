import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="🍊 Fruit Counter", layout="centered")
st.title("🍊 Fruit Counting App")

# Load trained YOLO model
model = YOLO("bestModel2.pt")  # Ensure path is correct

# Sidebar options
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
show_labels = st.sidebar.checkbox("Show Labels", value=True)
show_conf = st.sidebar.checkbox("Show Confidence", value=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Run detection with confidence threshold
    results = model.predict(image, conf=conf_threshold)

    # Count detected fruits
    count = len(results[0].boxes)

    # Plot results based on toggles
    plotted_img = results[0].plot(
        labels=show_labels,
        conf=show_conf
    )

    # Show result
    st.image(plotted_img, caption=f"Detected Fruits: {count}")
    st.success(f"Total Fruits Detected: {count}")


