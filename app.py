
import streamlit as st
import numpy as np
import keras
import numpy as np
import keras
import keras.utils as im
import matplotlib.pyplot as plt
from keras.models import Model
from keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import pickle

# Load the pre-trained model
model = load_model("Tea-LeavesDisease-Detection-Model.h5")

def predict(model, image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    x = im.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    output = np.argmax(model.predict(img_data), axis=1)
    
    index = ['Anthracnose', 'Algal Leaf', 'Bird Eye Spot', 'Brown Blight', 'Gray Light', 'Healthy', 'Red Leaf Spot', 'White Spot']
    result = index[output[0]]
    return result


def main():
    st.title("Tea Leaf Disease Detection System")
    
    page = st.sidebar.selectbox("Select a page", ["Model Prediction", "About the Model"])
    
    if page == "Model Prediction":
        model_prediction_page()

def model_prediction_page():
    st.header("Model Prediction")
    
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Making prediction..."):
            prediction = predict(model, image)

        st.subheader("Prediction Results")
        st.write(prediction)

if __name__ == "__main__":
    main()
