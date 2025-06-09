import animal_classification_CatDog

image = animal_classification_CatDog.load_image_file("test-animals/Gustav_chocolate.jpg")
result = animal_classification_CatDog.breed(image, image_path="test-animals/Gustav_chocolate.jpg")

print(result)
