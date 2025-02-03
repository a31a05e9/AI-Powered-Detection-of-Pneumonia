# Deep Breath - AI-Powered Pneumonia Detection

This project uses deep learning and a pre-trained model to detect pneumonia from chest X-ray images. The app is built using **Streamlit**, and users can upload an image to receive a prediction of whether the X-ray indicates pneumonia or not.

## Project Overview

- **Model**: The model used is a convolutional neural network (CNN) trained to classify X-ray images into two classes: pneumonia and normal.
- **Libraries**: The project uses `Keras`, `Streamlit`, `PIL`, and other Python libraries.
- **Web Application**: Built using Streamlit to provide an interactive user interface where users can upload X-ray images and get predictions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/deep-breath.git
    cd deep-breath
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Model

- The model used in this project is a trained `pneumonia_classifier.h5` file, which is a deep convolutional neural network for image classification.
- The class names are loaded from a `labels.txt` file, which maps indices to the pneumonia or normal label.

## Usage

1. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open the URL that Streamlit provides in your browser. It will look something like this: `http://localhost:8501`.

3. Upload a chest X-ray image (JPEG, PNG, JPG formats are supported).

4. The app will display the predicted class (`Pneumonia` or `Normal`) and a confidence score.

## Files Structure

- `app.py`: Main Streamlit app file.
- `util.py`: Utility functions for image processing and model classification.
- `model/`: Contains the trained model (`pneumonia_classifier.h5`) and label file (`labels.txt`).
- `bgs/`: Contains background images for the Streamlit app.

## Notes

- The model expects the image to be in a specific format and size. The app will resize images before sending them to the model.
- The app's performance may vary depending on the input image quality and the model's training data.

