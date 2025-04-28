import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import os

from util import classify

# Set the title and image
st.title('Pneumonia Detection')

image_path = os.path.join(os.path.dirname(__file__), "bgs/bg.jpg")
st.image(image_path, width=650)

# Upload file header
st.header('Please upload a chest X-ray image')

# File uploader
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load the model
model_path = r'C:\Users\Meher\Downloads\DeepBreath AI-Powered Detection of Pneumonia\DeepBreath AI-Powered Detection of Pneumonia\code\model\pneumonia_classifier.h5'
model = load_model(model_path)

# Load class names from labels.txt
labels_path = r'../model/labels.txt'
with open(labels_path, 'r') as f:
    class_names = [a.strip().split(' ')[1] for a in f.readlines()]

# If a file is uploaded
if file is not None:
    image = Image.open(file).convert('RGB')
    
    # Resize the image to the input size of the model (assuming 100x100)
    image = image.resize((100, 100), Image.LANCZOS)
    st.image(image, width=300)

    # Classify the image
    class_name, conf_score = classify(image, model, class_names)

    # Display the result
    st.write("## Predicted Class: {}".format(class_name))
    st.write("### Confidence Score: {}%".format(int(conf_score * 1000) / 10))
