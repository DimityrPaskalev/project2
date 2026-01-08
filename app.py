import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")

# Load & train model (cached)
@st.cache_resource
def load_model():
    try:
        from sklearn.datasets import load_digits
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split

        digits = load_digits()
        X = digits.images.reshape(len(digits.images), -1) / 16.0
        y = digits.target

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


model = load_model()

if model is None:
    st.warning("Could not load model.")
else:
    st.success("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None and model is not None:
    try:
        # Display image
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=200)

        # Resize to 8x8 (digits dataset size)
        img_resized = image.resize((8, 8))
        img_array = np.array(img_resized)

        # Invert colors if background is light
        if np.mean(img_array) > 128:
            img_array = 255 - img_array

        # Normalize
        img_array = img_array / 16.0
        img_flat = img_array.reshape(1, -1)

        # Prediction
        prediction = model.predict(img_flat)[0]
        st.write(f"## 🧠 Prediction: **{prediction}**")

        # Probabilities
        probs = model.predict_proba(img_flat)[0]
        st.write("### Probabilities:")
        for i, prob in enumerate(probs):
            st.write(f"Digit {i}: {prob:.2%}")

    except Exception as e:
        st.error(f"Error processing image: {e}")

elif uploaded_file is not None:
    st.warning("Model not available, cannot predict.")

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a handwritten digit (0–9)
2. Image is resized to 8x8 pixels
3. AI predicts the digit
4. Best results:
   - White background
   - Black digit
   - Centered
   - Minimal noise
""")
