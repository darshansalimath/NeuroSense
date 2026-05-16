import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Potato Leaf Disease Detection", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    h1 {
        color: #2e8b57;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        text-shadow: 1px 1px 2px #d3d3d3;
    }
    .stUpload {
        border-radius: 15px;
        padding: 10px;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 20px;
    }
    .disease-name {
        font-size: 24px;
        font-weight: bold;
        color: #d9534f;
    }
    .healthy-name {
        font-size: 24px;
        font-weight: bold;
        color: #5cb85c;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🥔 Potato Leaf Disease Detection")
st.write("Upload a clear image of a potato leaf to detect if it is healthy or infected with Early or Late Blight.")

MODEL_PATH = 'potato_disease_model.h5'
CLASSES_PATH = 'class_names.txt'
IMAGE_SIZE = (256, 256)

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

@st.cache_data
def load_class_names():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]
    # Default fallback
    return ["Early_Blight", "Healthy", "Late_Blight"]

model = load_trained_model()
class_names = load_class_names()

if model is None:
    st.error(f"⚠️ Model not found at `{MODEL_PATH}`. Please run `train_model.py` first to train the model, or place your trained `.h5` file in this directory.")
else:
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses a Convolutional Neural Network (CNN) "
        "to classify potato leaf diseases into three categories:\n"
        "- **Healthy**\n"
        "- **Early Blight**\n"
        "- **Late Blight**\n\n"
        "Built with TensorFlow, Keras, and Streamlit."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write("### How to use:")
    st.sidebar.write("1. Take a close-up photo of a potato leaf.")
    st.sidebar.write("2. Upload the image using the browse button.")
    st.sidebar.write("3. View the prediction results below the image.")

    # Main Content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Leaf Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Preprocessing
            st.write("Processing image...")
            # Convert to RGB (in case of RGBA)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            img_resized = image.resize(IMAGE_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            # The Rescaling layer (1./255) is already inside our model, 
            # so we pass the raw pixel values directly to the model.

            # Prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            predicted_class = class_names[predicted_class_index]

            with col2:
                st.subheader("Analysis Result")
                
                # Dynamic styling based on prediction
                if "healthy" in predicted_class.lower():
                    st.markdown(f"""
                        <div class="prediction-box">
                            <p>Prediction:</p>
                            <div class="healthy-name">{predicted_class.replace('_', ' ')}</div>
                            <p>Confidence: {confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("The leaf appears to be healthy! No treatment needed.")
                else:
                    st.markdown(f"""
                        <div class="prediction-box">
                            <p>Prediction:</p>
                            <div class="disease-name">{predicted_class.replace('_', ' ')}</div>
                            <p>Confidence: {confidence:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.warning(f"Disease Detected: {predicted_class.replace('_', ' ')}. Please consult agricultural guidelines for treatment.")

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

