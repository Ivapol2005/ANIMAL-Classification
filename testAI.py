import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from code.breeds_list import breeds, cat_breeds, dog_breeds

model = tf.keras.models.load_model("cat_dog_classifier.h5")

IMG_SIZE = (128, 128)

# cat_breeds = {1, 6, 7, 8, 10, 12, 21, 24, 27, 28, 33, 34}
# dog_breeds = {2, 3, 4, 5, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 29, 30, 31, 32, 35, 36, 37}

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = np.array(image) / 255.0
    return image

def visualize_classification(image_path, predicted_class, confidence, predicted_animal):
    image = cv2.imread(image_path)
    if image is None:
        print(f"can't open {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    class_name = breeds.get(predicted_class, "Unknown")
    animal_type = "Cat" if predicted_animal == 0 else "Dog"

    cv2.rectangle(image, (5, 5), (w - 5, h - 5), (255, 0, 0), 2)
    cv2.putText(image, f"{animal_type}, {class_name} ({confidence:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    print(f"{os.path.basename(image_path)} → {animal_type}, {class_name} ({confidence*100:.2f}%)")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

"""
for image_file in image_files:
    image_path = os.path.join(test_folder, image_file)
    processed_image = preprocess_image(image_path)

    if processed_image is not None:
        input_image = np.expand_dims(processed_image, axis=0)
        prediction_breed, prediction_animal = model.predict(input_image)

        predicted_class = np.argmax(prediction_breed)
        confidence = np.max(prediction_breed)

        predicted_animal = 0 if predicted_class in cat_breeds else 1

        visualize_classification(image_path, predicted_class, confidence, predicted_animal)
    else:
        print(f"Skipped: {image_file}")
"""

