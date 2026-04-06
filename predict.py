import numpy as np
import cv2
from tensorflow.keras.models import load_model

# load saved model
model = load_model("models/car_detection_resnet50.h5")

print("Model loaded")

def predict_image(img_path):

    image = cv2.imread(img_path)
    display = image.copy()

    image = cv2.resize(image,(224,224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        label = "Car"
    else:
        label = "No Car"

    print("Prediction:", label)
    print("Confidence:", prediction)

    cv2.putText(display,label,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Prediction",display)
    cv2.waitKey(0)

