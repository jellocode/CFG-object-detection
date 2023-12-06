import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# function to create label(dictionary)
def load_datasets(dataset_categories):
    images = []
    labels = []

    for dataset_id, (dataset_path, category) in enumerate(dataset_categories.items()):
        for object_category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, object_category)

            for file in os.listdir(category_path):
                img_path = os.path.join(category_path, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))  # resize all images
                images.append(img)
                labels.append({
                    'category': category,
                    'dataset_id': dataset_id
                })

    return images, labels

# loading datasets

# define dataset paths and corresponding category names
dataset_categories = {
    r'images\bee_image_sample': 'bee',
    r'images\football_image_sample': 'football',
    r'images\keyboard_image_sample': 'keyboard',
    r'images\laptop_image_sample': 'laptop',
    r'images\Letter_M_image_sample': 'letter M',
    r'images\Letter_T_image_sample': 'letter T',
    r'images\monitor_image_sample': 'monitor',
    r'images\mouse-image-sample': 'mouse',
    r'images\trains_image_sample': 'trains',
}

# load and preprocess the datasets

images, labels = load_datasets(dataset_categories)
print(images,labels)

# convert labels to numerical format
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform([label['category'] for label in labels])
encoded_labels = to_categorical(encoded_labels)

# split the dataset into training, testing sets
X_train, X_test, y_train, y_test = train_test_split(np.array(images), encoded_labels, test_size=0.2, random_state=42)

print("Number of Datasets:", len(dataset_categories))

# verifying dataset is loaded correctly 
for dataset_path, category in dataset_categories.items():
    num_images = sum(len(files) for _, _, files in os.walk(dataset_path))
    print(f"dataset: {dataset_path}, category: {category}, no. of images: {num_images}")
