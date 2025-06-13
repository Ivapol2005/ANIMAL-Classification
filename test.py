import animal_classification_CatDog
import os

test_folder = "test-animals"
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(test_folder, image_file)

    image = animal_classification_CatDog.load_image_file(image_path)
    result = animal_classification_CatDog.breed(image, image_path=image_path, show_visualization=True, all_predictions=True)

    print(f"Predicted Animal: {result['predicted_animal'].capitalize()}")
    print(f"Top Breed: {result['top_breed']['name']} ({result['top_breed']['confidence']:.2%})")

    if 'all_predictions' in result and result['all_predictions']:
        print("\n--- All Top Predictions ---")
        for i, pred in enumerate(result['all_predictions'][:5]):
            print(f"  {i+1}. {pred['name']} ({pred['confidence']:.2%}) {'(Cat)' if pred['is_cat'] else '(Dog)'}")