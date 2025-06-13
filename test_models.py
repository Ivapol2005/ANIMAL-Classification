import animal_classification_CatDog
import os
import numpy as np
from src.breeds_list import breeds, cat_breeds, dog_breeds

test_folder = "../dataset/images"
results = []

image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_true_info_from_filename(filename):
    filename_lower = filename.lower()
    
    for breed_id, breed_name in breeds.items():
        if breed_name.lower() in filename_lower:
            true_animal_type = None
            if breed_id in cat_breeds:
                true_animal_type = "cat"
            elif breed_id in dog_breeds:
                true_animal_type = "dog"
            
            if true_animal_type:
                return true_animal_type, breed_name

    return None, None


def check_breed_name_match(filename_true_breed_name, predicted_breed_name):
    if filename_true_breed_name is None or predicted_breed_name is None:
        return False

    filename_lower = filename_true_breed_name.lower()
    predicted_lower = predicted_breed_name.lower()

    return (predicted_lower == filename_lower or
            predicted_lower in filename_lower or
            filename_lower in predicted_lower)


for image_file in image_files:
    image_path = os.path.join(test_folder, image_file)
    
    true_animal_type, true_breed_name = get_true_info_from_filename(image_file)
    
    if true_animal_type is None:
        print(f"SKIPPED: Could not determine true animal/breed from filename for {image_file}. Skipping.")
        continue

    try:
        image_processed = animal_classification_CatDog.load_image_file(image_path)
        if image_processed is not None:
            prediction_result = animal_classification_CatDog.breed(image_processed, image_path=image_path, show_visualization=False)
            
            prediction_result['image_filename'] = image_file
            prediction_result['true_animal'] = true_animal_type
            prediction_result['true_breed'] = true_breed_name
            
            breed_name_match = check_breed_name_match(true_breed_name, prediction_result['top_breed']['name'])
            prediction_result['breed_name_match'] = breed_name_match

            animal_type_match = (prediction_result['predicted_animal'] == true_animal_type)
            prediction_result['animal_type_match'] = animal_type_match

            results.append(prediction_result)
            
            status_breed = "OK" if breed_name_match else "MISMATCH"
            status_animal = "OK" if animal_type_match else "MISMATCH"
            
            print(f"Processed {image_file}: True Animal: {true_animal_type}, True Breed: {true_breed_name}")
            print(f"  -> Predicted Animal: {prediction_result['predicted_animal']} ({status_animal}), Predicted Breed: {prediction_result['top_breed']['name']} ({status_breed})")
            print(f"  Confidence: {prediction_result['top_breed']['confidence']:.2%}")
        else:
            print(f"Failed to load {image_file}")
    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")

print("\n--- Evaluation Summary by Category ---")

# Stats by breed
category_stats = {}
for res in results:
    true_animal = res['true_animal']
    if true_animal not in category_stats:
        category_stats[true_animal] = {'total': 0, 'correct_animal_matches': 0, 'correct_breed_matches': 0}
    
    category_stats[true_animal]['total'] += 1
    if res['animal_type_match']:
        category_stats[true_animal]['correct_animal_matches'] += 1
    if res['breed_name_match']:
        category_stats[true_animal]['correct_breed_matches'] += 1

total_overall_processed = 0
total_overall_correct_animal = 0
total_overall_correct_breed = 0

for animal_type, stats in category_stats.items():
    total_in_category = stats['total']
    correct_animal_in_category = stats['correct_animal_matches']
    correct_breed_in_category = stats['correct_breed_matches']
    
    animal_accuracy = (correct_animal_in_category / total_in_category) * 100 if total_in_category > 0 else 0
    breed_accuracy = (correct_breed_in_category / total_in_category) * 100 if total_in_category > 0 else 0
    
    total_overall_processed += total_in_category
    total_overall_correct_animal += correct_animal_in_category
    total_overall_correct_breed += correct_breed_in_category

    print(f"\nCategory: {animal_type.capitalize()}")
    print(f"  Total {animal_type} images: {total_in_category}")
    print(f"  Correct Animal Type Predictions: {correct_animal_in_category} ({animal_accuracy:.2f}%)")
    print(f"  Correct Breed Name Matches: {correct_breed_in_category} ({breed_accuracy:.2f}%)")

# Overall stats
overall_animal_accuracy = (total_overall_correct_animal / total_overall_processed) * 100 if total_overall_processed > 0 else 0
overall_breed_accuracy = (total_overall_correct_breed / total_overall_processed) * 100 if total_overall_processed > 0 else 0

print(f"\n--- Overall Summary ---")
print(f"Total images processed (overall): {total_overall_processed}")
print(f"Overall Correct Animal Type Predictions: {total_overall_correct_animal} ({overall_animal_accuracy:.2f}%)")
print(f"Overall Correct Breed Name Matches: {total_overall_correct_breed} ({overall_breed_accuracy:.2f}%)")

print("\n--- Detailed Results List ---")
for i, res in enumerate(results, 1):
    status_animal = "OK" if res['animal_type_match'] else "MISMATCH"
    status_breed = "OK" if res['breed_name_match'] else "MISMATCH"
    
    print(f"{i}. File: {res['image_filename']}")
    print(f"   True: {res['true_animal'].capitalize()} / {res['true_breed']}")
    print(f"   Pred: {res['predicted_animal'].capitalize()} ({status_animal}) / {res['top_breed']['name']} ({status_breed})")
    print(f"   Confidence: {res['top_breed']['confidence']:.2%}")
    print("-" * 30)