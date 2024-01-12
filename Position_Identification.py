#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:02:00 2024

@author: arpanrajpurohit
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

# Set the paths to your data folders/files
image_folder = 'Screenshots/'
vector_csv_path = 'Screenshots/Positions.csv'

# Load image data
num_images = 8000
image_filenames = [f"screenshot_{i}.png" for i in range(num_images)]
image_data = []

for filename in image_filenames[:num_images]:
    image_path = os.path.join(image_folder, filename)
    image = cv2.resize(np.array(Image.open(image_path).convert('L')), (64, 64))
    image_data.append(image)

image_data = np.array(image_data)

# Load vector data
vector_df = pd.read_csv(vector_csv_path, header=None, skiprows=2).drop(3, axis=1).values.astype('float32')


num_samples = 5
# Create a figure
fig = plt.figure(figsize=(15, 5))
# Set the overall figure title
fig.suptitle('Sample Images with Cube Locations', fontsize=16)
# Plot the images
for i in range(num_samples):
    # Create subplots
    plt.subplot(1, num_samples, i + 1)  
    # Plot image
    plt.imshow(image_data[i], cmap='gray') 
    # Set subplot title
    plt.title(f"Cube Location {vector_df[i]}", fontsize=12)  
    # Turn off axis labels
    plt.axis('off')
# Adjust layout for better spacing
plt.tight_layout()
# Show the plot
plt.show()

# Split data into train and test sets
trainX, testX, trainY, testY = train_test_split(image_data, vector_df, test_size=0.2, random_state=42, shuffle=True)

# Normalize pixels
def normalize_pixels(train, test):
    return train.astype('float32') / 255.0, test.astype('float32') / 255.0

trainX, testX = normalize_pixels(trainX, testX)

# Normalize data
def normalize_data(train_data, test_data):
    min_val, max_val = np.min(train_data), np.max(train_data)
    range_val = max_val - min_val
    return ((train_data - min_val) / range_val).astype(np.float32), ((test_data - min_val) / range_val).astype(np.float32)

trainY, testY = normalize_data(trainY, testY)

# Define the model
def build_model():
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='linear')
    ])

    model.compile(optimizer=Adam(), loss='mse', metrics=['mae', 'accuracy'])
    model.summary()
    return model

# Train the model
epochs, batch_size = 10, 32
model = build_model()
history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
evaluation = model.evaluate(testX, testY)

# Print final metrics
print("Evaluation Metrics:")
print(" - Loss:", evaluation[0])
print(" - Mean Absolute Error (MAE):", evaluation[1])
print(" - Accuracy:", evaluation[2])

# Save the model
model.save("position_prediction_model.h5")

# Make a prediction on a single image
single_image = np.expand_dims(testX[0], axis=0)
normalized_prediction = model.predict(single_image)

# Denormalize the prediction and output
def denormalize_array(normalized_arr, original_min, original_max):
    range_val = original_max - original_min
    return (normalized_arr * range_val + original_min).astype(np.float32)

prediction = denormalize_array(normalized_prediction, np.min(vector_df), np.max(vector_df))
output = denormalize_array(testY[0], np.min(vector_df), np.max(vector_df))

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(6, 4))
# Plot the original image on the left subplot
axs[0].imshow(single_image[0], cmap='gray')
axs[0].axis('off')
axs[0].set_title('Original Image')
# Display the plain values on the right subplot
axs[1].text(0.5, 0.5, f"Output Values:\n{output}\n\nPredicted Values:\n{np.round(prediction[0],2)}", 
            va='center', ha='center', fontsize=12)
axs[1].axis('off')
axs[1].set_title('Output and Predicted Values')
# Adjust layout to prevent clipping
plt.tight_layout()
# Show the plots
plt.show()

