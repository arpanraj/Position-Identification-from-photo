# Cube Position Identification from Screenshot

This repository contains Python code for a research project focused on identifying the positions of cubes in screenshots. The project involves using image processing techniques and machine learning to achieve accurate cube location predictions.

## Overview

The code is organized into several sections:

1. **Data Loading**: The script loads image data from a specified folder and vector data from a CSV file. The images are resized to 64x64 pixels and converted to grayscale.

2. **Data Visualization**: A sample of images with corresponding cube locations is visualized to provide insight into the dataset.

3. **Data Preprocessing**: The image and vector data are split into training and testing sets. Pixel values of images are normalized, and vector data is normalized to a range between 0 and 1.

4. **Model Architecture**: A convolutional neural network (CNN) model is defined using TensorFlow's Keras API. The model is designed to predict the x, y, and z coordinates of cube positions.

5. **Model Training**: The defined model is trained on the preprocessed data. Training progress is visualized through plots of accuracy and loss.

6. **Model Evaluation**: The trained model is evaluated on the test set, and metrics such as loss, mean absolute error (MAE), and accuracy are printed.

7. **Model Saving**: The trained model is saved to a file named "position_prediction_model.h5" for later use.

8. **Prediction Example**: The script demonstrates making predictions on a single test image and visualizing the original image alongside the predicted and actual cube positions.

## Usage

To use this code, follow these steps:

1. Set the paths to your image folder and vector CSV file in the script.

```python
image_folder = '/path/to/your/image/folder/'
vector_csv_path = '/path/to/your/vector/file.csv'