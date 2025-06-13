# animal_classification_CatDog.py

import numpy as np
from src.testAI import preprocess_image, visualize_classification, model
from src.breeds_list import breeds, cat_breeds, dog_breeds
import os
from PIL import Image

cat_breeds_ids = set(cat_breeds.keys())
dog_breeds_ids = set(dog_breeds.keys())

def load_image_file(image_path):
    processed = preprocess_image(image_path)
    if processed is not None:
        return np.expand_dims(processed, axis=0)
    else:
        raise ValueError(f"Image {image_path} could not be processed.")

def breed(input_image, image_path=None, show_visualization=False, all_predictions=False):

    predictions = model.predict(input_image)
    
    breed_probs = predictions['breed_output'][0]
    animal_prob = predictions['animal_output'][0][0]
    
    is_dog = animal_prob > 0.5

    breed_probs_filtered = np.zeros_like(breed_probs)
    
    if is_dog:
        predicted_animal_str = "Dog"
        target_breed_ids_set = set(dog_breeds.keys())
    else:
        predicted_animal_str = "Cat"
        target_breed_ids_set = set(cat_breeds.keys())

    for i, prob in enumerate(breed_probs):
        current_breed_id = i
        if current_breed_id in target_breed_ids_set:
            breed_probs_filtered[i] = prob

    top1_idx_in_probs = np.argmax(breed_probs_filtered)
    top1_conf = float(breed_probs_filtered[top1_idx_in_probs])

    predicted_breed_id = top1_idx_in_probs

    breed1 = breeds.get(predicted_breed_id, "Unknown")

    result = {
        "top_breed": {
            "name": breed1,
            "confidence": top1_conf
        },
        "predicted_animal": predicted_animal_str,
    }

    if all_predictions:
        all_breeds_detailed = []
        for i, prob in enumerate(breed_probs.tolist()):
            breed_id_for_all = i
            breed_name_for_all = breeds.get(breed_id_for_all, "Unknown")
            is_cat_for_all = (breed_id_for_all in cat_breeds_ids)
            
            all_breeds_detailed.append({
                "id": breed_id_for_all,
                "name": breed_name_for_all,
                "confidence": float(prob),
                "is_cat": is_cat_for_all
            })
        
        all_breeds_detailed_sorted = sorted(all_breeds_detailed, key=lambda x: x['confidence'], reverse=True)
        result["all_predictions"] = all_breeds_detailed_sorted

    if show_visualization and image_path:
        visualize_classification(image_path, predicted_breed_id, top1_conf, predicted_animal_str)
    
    return result