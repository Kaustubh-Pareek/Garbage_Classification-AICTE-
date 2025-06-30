import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import time

# load the model
model = tf.keras.models.load_model(r'week2\Effiicientnetv2b2.keras')
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


with open('recycle_info.json', 'r') as f:
    recycle_info = json.load(f)

# function for preprocessing the image
def preprocess_image(image):
    image = image.resize((124, 124))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# function for prediction
def predict_image(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    pred_class = class_names[np.argmax(prediction)]
    return pred_class


st.set_page_config(page_title="Garbage Classifier", layout="centered")

page = st.sidebar.selectbox("Choose a page", ["Home", "Garbage Classifier"])

# home page
if page == "Home":
    st.title("♻️ Smart Garbage Classification")
    st.markdown("""
    Welcome to the Smart Garbage Classifier powered by Deep Learning.

    This project uses a cutting-edge **EfficientNetV2B2** model to automatically classify waste into categories such as **biodegradable**, **non-biodegradable**, and **recyclable**, promoting efficient waste management and 
    recycling.

    ### How it works:
    🔍 Upload an image of waste material  
    ⚙️ The model predicts the waste type (e.g., glass, plastic, etc.)  
    🌱 It suggests useful tips on how to recycle it  

    👉 Go to the **Garbage Classifier** page from the sidebar to try it out!
    """)

#garbage classifier page
elif page == "Garbage Classifier":
    st.title("🖼️ Upload Waste Image for Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)  

        
        if st.button("🚀 Predict"):
            
            with st.spinner("Launching rocket to classify... 🚀"):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                predicted_class = predict_image(image)
            st.success(f"✅ Predicted Class: **{predicted_class.capitalize()}**")

            
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Your Image", width=250)

            with col2:
                st.markdown("### ♻️ How to Recycle")
                st.info(recycle_info.get(predicted_class, "No info available."))
