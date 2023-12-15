import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# load the model
loaded_model = load_model('object_recognition_model.h5')

# load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy')

# preprocess new image
new_image_path = 'test.jpg'
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (224, 224))

# match the input shape
new_image = np.expand_dims(new_image, axis=0)

# make predictions
predictions = loaded_model.predict(new_image)

# get the predicted class
predicted_class_index = np.argmax(predictions[0])
predicted_class = label_encoder.classes_[predicted_class_index]

print(f"model prediction : image belongs to class {predicted_class}")
