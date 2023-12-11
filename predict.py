import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# load the model
loaded_model = load_model('object_recognition_model.h5')

# Assuming you saved the label encoder during training
# If not, you need to load or recreate the label encoder with the same configuration used during training
# For example, if you saved it during training like this:
# joblib.dump(label_encoder, 'label_encoder.pkl')
# You would load it here like this:
# label_encoder = joblib.load('label_encoder.pkl')

# load the label encoder (replace 'label_encoder.pkl' with the actual path if saved during training)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy')

# preprocess a new image
new_image_path = 'blank.jpg'
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (224, 224))  # Resize to the same size used during training

# match the input shape
new_image = np.expand_dims(new_image, axis=0)

# make predictions
predictions = loaded_model.predict(new_image)

# get the predicted class
predicted_class_index = np.argmax(predictions[0])
predicted_class = label_encoder.classes_[predicted_class_index]

print(f"The model predicts that the image belongs to class: {predicted_class}")
