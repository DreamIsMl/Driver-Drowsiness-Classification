import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image  # Import the Image module

# Load your model here
# Replace 'your_model_path' with the actual path to your model file
model = tf.keras.models.load_model('F:\\class\\DeepLearning\\Driver Drowsiness Classification\\model (1).h5')
IMAGE_SIZE = [255, 255]

def predict(model, img_path, class_names):
    image = Image.open(img_path)  # Open the image using PIL
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image)
    img_batch = np.expand_dims(img_array, 0)

    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

# Replace 'class_names' with your actual class names
class_names = ['Closed', 'Open', 'no_yawn', 'yawn']

# Streamlit UI
st.title("Image Prediction App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Make a prediction when the button is clicked
    if st.button("Predict"):
        # Save the uploaded file to a temporary location
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Make a prediction using the model
        result_class, result_confidence = predict(model, img_path, class_names)

        # Display the prediction result
        st.success(f"Predicted Class: {result_class}")
        st.success(f"Confidence: {result_confidence}%")
