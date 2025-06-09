import testAI
from testAI import preprocess_image, visualize_classification, model
import os
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from breeds_list import *

# Приховуємо головне вікно Tk
Tk().withdraw()

# Вибір файлів через діалог
image_files = askopenfilenames(
    title="Виберіть зображення для класифікації",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

# Опрацьовуємо кожне зображення
for image_path in image_files:
    processed_image = preprocess_image(image_path)

    if processed_image is not None:
        input_image = np.expand_dims(processed_image, axis=0)
        prediction_breed, prediction_animal = model.predict(input_image)

        # Top-2 породи
        top2_indices = np.argsort(prediction_breed[0])[-2:][::-1]
        top1_idx, top2_idx = top2_indices
        top1_conf = prediction_breed[0][top1_idx]
        top2_conf = prediction_breed[0][top2_idx]

        predicted_animal = 0 if top1_idx in cat_breeds else 1

        visualize_classification(image_path, top1_idx, top1_conf, predicted_animal)

        breed1 = breeds.get(top1_idx, "Unknown")
        breed2 = breeds.get(top2_idx, "Unknown")

        # if abs(top1_conf - top2_conf) < 0.1:
        print(f"Possible breeds: {breed1} ({top1_conf:.2f}) or {breed2} ({top2_conf:.2f})")
        # else:
            # print(f"Predicted breed: {breed1} (Confidence: {top1_conf:.2f})")
    else:
        print(f"Skipped: {os.path.basename(image_path)}")
