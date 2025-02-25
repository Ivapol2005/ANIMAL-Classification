import nlp
from nlp import text_input
import testAI
from testAI import preprocess_image, visualize_classification, model
import os
import  numpy as np
from code.breeds_list import *

test_folder = "test-animals"
image_files = [f for f in os.listdir(test_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    text = input("Enter a sentence: ")
    breed_quess = text_input(text)

    image_path = os.path.join(test_folder, image_file)
    processed_image = preprocess_image(image_path)

    if processed_image is not None:
        input_image = np.expand_dims(processed_image, axis=0)
        prediction_breed, prediction_animal = model.predict(input_image)

        predicted_class = np.argmax(prediction_breed)
        confidence = np.max(prediction_breed)

        predicted_animal = 0 if predicted_class in cat_breeds else 1

        visualize_classification(image_path, predicted_class, confidence, predicted_animal)
        
        predicted_breed_name = breeds.get(predicted_class, "Unknown").lower()
        breed_quess_lower = [b.lower() for b in breed_quess]
        if predicted_breed_name in breed_quess_lower:
            print("correct")
        else:
            print(f"❌ Incorrect. Expected: {breed_quess}, Predicted: {predicted_breed_name}")
    else:
        print(f"Skipped: {image_file}")
