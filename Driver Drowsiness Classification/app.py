import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load the saved model
model_path = 'F:\\class\\DeepLearning\\Driver Drowsiness Classification\\model (1).h5'
model = tf.keras.models.load_model(model_path)

BATCH_SIZE = 32
IMAGE_SIZE = 255
CHANNELS = 3

class_names = ['Closed', 'Open', 'no_yawn', 'yawn']

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0 / 255)
])

# Load the Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the input frame
def preprocess_frame(frame):
    resized_frame = resize_and_rescale(frame)
    input_frame = tf.expand_dims(resized_frame, axis=0)
    return input_frame

# Function to predict using the model
def predict(model, input_frame):
    predictions = model.predict(input_frame)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Load the image
image_path = 'F:\\class\\DeepLearning\\Driver Drowsiness Classification\\1007.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Extract the face region for prediction
    face_region = image[y:y + h, x:x + w]

    # Preprocess the face region
    input_frame = preprocess_frame(face_region)

    # Make prediction
    predicted_class, confidence = predict(model, input_frame)

    # Display the prediction on the image
    cv2.putText(image, f"Predicted: {predicted_class} ({confidence:.2f}%)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Display the image with predictions
cv2.imshow('Image Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
