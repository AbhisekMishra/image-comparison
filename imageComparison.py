import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import cv2
# import loadModel

# Load a pre-trained VGG16 model (you can use other models depending on your requirements)
# model = VGG16(weights='imagenet')
model = tf.keras.models.load_model('custom_model')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize images to the desired input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)[0][0]
    # decoded_predictions = decode_predictions(predictions)
    return predictions

# Example usage: Load and predict an image
cheque_image_path = 'image_sample/standard_chartered.png'
prediction = predict_image(cheque_image_path)
is_genuine = prediction < 0.5
print(f"Prediction for {cheque_image_path} is {prediction} which is Genuine" if is_genuine else f"Prediction: {prediction} which is Forged")
print(f"Confidence: {1 - prediction if is_genuine else prediction:.4f}\n")
