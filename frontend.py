import streamlit as st
import requests
from PIL import Image

# =========================
# API URL (Docker FastAPI)
# =========================
API_URL = "http://localhost:8000/predict"

# =========================
# Disease Knowledge Base
# =========================
disease_info = {
    "Tomato___Early_blight": {
        "cure": "Remove infected leaves and use copper-based fungicides.",
        "prevention": "Avoid overhead watering and maintain proper spacing between plants."
    },
    "Tomato___Late_blight": {
        "cure": "Use fungicides like chlorothalonil or mancozeb immediately.",
        "prevention": "Ensure good drainage and avoid wet leaves for long periods."
    },
    "Tomato___healthy": {
        "cure": "No treatment needed.",
        "prevention": "Continue regular watering and good farming practices."
    }
}

# =========================
# UI Title
# =========================
st.title("🌿 Crop Disease Detection System")
st.write("Upload a plant leaf image to detect disease and get treatment advice")

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict Disease"):

        with st.spinner("Analyzing image..."):

            files = {"file": uploaded_file.getvalue()}

            response = requests.post(API_URL, files=files)

        # =========================
        # Display Results
        # =========================
        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            confidence = result["confidence"]

            st.success(f"Prediction: {prediction}")
            st.info(f"Confidence: {confidence} %")

            # =========================
            # Disease Info Section
            # =========================
            info = disease_info.get(prediction)

            if info:

                st.subheader("🌿 Cure / Treatment")
                st.write(info["cure"])

                st.subheader("🛡️ Prevention Tips")
                st.write(info["prevention"])

            else:
                st.warning("No treatment info available for this class.")

        else:
            st.error("Failed to connect to API. Make sure Docker is running.")