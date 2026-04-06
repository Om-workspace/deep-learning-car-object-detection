import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

print("Loading dataset...")

DATASET_PATH = "dataset"

images = []
labels = []

# --------------------
# Load car images
# --------------------
car_path = os.path.join(DATASET_PATH, "training_images")

for img in os.listdir(car_path):
    img_path = os.path.join(car_path, img)

    image = cv2.imread(img_path)
    if image is None:
        continue

    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32) / 255.0

    images.append(image)
    labels.append(1)

# --------------------
# Load non car images
# --------------------
non_car_path = os.path.join(DATASET_PATH, "non_cars")

if os.path.exists(non_car_path):
    for img in os.listdir(non_car_path):
        img_path = os.path.join(non_car_path, img)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.resize(image, (224, 224))
        image = np.array(image, dtype=np.float32) / 255.0

        images.append(image)
        labels.append(0)

X = np.array(images)
y = np.array(labels)

print("Dataset Loaded")
print("Shape:", X.shape)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

import gc
del X, y, images
try:
    del image
except:
    pass
gc.collect()

# ----------------------------
# function to build model
# ----------------------------
def build_model():

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


# ----------------------------
# Adam
# ----------------------------
print("Training with Adam optimizer...")
model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_adam = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# ----------------------------
# SGD
# ----------------------------
from tensorflow.keras.optimizers import SGD

print("Training with SGD optimizer...")
model = build_model()

model.compile(
    optimizer=SGD(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_sgd = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# ----------------------------
# RMSprop
# ----------------------------
print("Training with RMSprop optimizer...")
model = build_model()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_rms = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# ----------------------------
# Results
# ----------------------------
print("\nFinal Accuracy Comparison")

print("Adam Accuracy:",
      max(history_adam.history['val_accuracy']))

print("SGD Accuracy:",
      max(history_sgd.history['val_accuracy']))

print("RMSprop Accuracy:",
      max(history_rms.history['val_accuracy']))

model.save("models/car_detection_resnet50.h5")
print("Model saved")

import json
metrics_data = {
    'epochs': list(range(1, len(history_adam.history['val_accuracy']) + 1)),
    'adam': history_adam.history['val_accuracy'],
    'sgd': history_sgd.history['val_accuracy'],
    'rmsprop': history_rms.history['val_accuracy']
}
os.makedirs("static", exist_ok=True)
with open("static/metrics.json", "w") as f:
    json.dump(metrics_data, f)
print("Metrics successfully exported to dashboard!")

plt.plot(history_adam.history['val_accuracy'], label='Adam')
plt.plot(history_sgd.history['val_accuracy'], label='SGD')
plt.plot(history_rms.history['val_accuracy'], label='RMSprop')

plt.title('Optimizer Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()