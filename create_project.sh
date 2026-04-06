#!/bin/bash

echo "Creating Car Object Detection Project Structure..."

# Root folder
PROJECT_NAME="car-object-detection-resnet50"

mkdir -p $PROJECT_NAME

cd $PROJECT_NAME

# Create folders
mkdir -p dataset
mkdir -p src
mkdir -p models
mkdir -p results

# Create files
touch main.py
touch README.md
touch requirements.txt

# Add starter main.py
cat <<EOL > main.py
import numpy as np
import os
import cv2
import tensorflow as tf

print("Car Object Detection using ResNet50")
EOL

# Add requirements
cat <<EOL > requirements.txt
tensorflow
keras
numpy
matplotlib
opencv-python
scikit-learn
pandas
EOL

echo "Project created successfully!"