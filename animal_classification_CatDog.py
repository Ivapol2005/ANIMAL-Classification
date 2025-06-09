# animal_classification_CatDog.py

import numpy as np
from src.testAI import preprocess_image, visualize_classification, model
from src.breeds_list import *
import os
from PIL import Image

def load_image_file(image_path):
    processed = preprocess_image(image_path)
    if processed is not None:
        return np.expand_dims(processed, axis=0)
    else:
        raise ValueError(f"Image {image_path} could not be processed.")

def breed(input_image, show_visualization=True, image_path=None):
    prediction_breed, prediction_animal = model.predict(input_image)

    top2_indices = np.argsort(prediction_breed[0])[-2:][::-1]
    top1_idx, top2_idx = top2_indices
    top1_conf = prediction_breed[0][top1_idx]
    top2_conf = prediction_breed[0][top2_idx]

    predicted_animal = 0 if top1_idx in cat_breeds else 1

    if show_visualization and image_path:
        visualize_classification(image_path, top1_idx, top1_conf, predicted_animal)

    breed1 = breeds.get(top1_idx, "Unknown")
    breed2 = breeds.get(top2_idx, "Unknown")

    """
    if abs(top1_conf - top2_conf) < 0.05:
        return f"Possible breeds: {breed1} ({top1_conf:.2f}) or {breed2} ({top2_conf:.2f})"
    else:
    """
    return f"{breed1} (Confidence: {top1_conf:.2f})"