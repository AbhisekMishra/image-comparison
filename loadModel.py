import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification: genuine or forged
    ])

    return model


def load_data(dataset_dir):
    images = []
    labels = []
    for label, class_dir in enumerate(['genuine', 'forged']):
        class_path = os.path.join(dataset_dir, class_dir)
        image_files = os.listdir(class_path)
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))  # Resize images to the desired input size
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def load_model():
    # Load the synthetic dataset
    dataset_dir = "dataset"
    images, labels = load_data(dataset_dir)

    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0

    # Build the custom model
    input_shape = x_train[0].shape
    model = build_custom_model(input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on the synthetic dataset
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
    model.save("custom_model")

load_model()