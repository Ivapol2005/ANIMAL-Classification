from code.breeds_list import breeds, cat_breeds, dog_breeds
from tensorflow.keras import layers, models
# from code.create_dataset import parse_voc_annotation, load_voc_dataset
# from code.model_settings import load_and_preprocess, tf_load_and_preprocess, create_tf_dataset
import pandas as pd
import numpy as np
import code.model_settings

dataset = []

dataset_test = []

from code.create_dataset import train_ds, test_ds

# import code.model_settings
from code.model_settings import model, early_stop, reduce_lr

history = model.fit(train_ds, validation_data=test_ds, epochs=20, callbacks=[early_stop, reduce_lr])

model.save("cat_dog_classifier.h5")


# metrics
from sklearn.metrics import classification_report

y_true_breed = []
y_true_animal = []

for batch in test_ds:
    labels = batch[1]
    y_true_breed.append(labels["breed_output"].numpy())
    y_true_animal.append(labels["animal_output"].numpy())

y_true_breed = np.concatenate(y_true_breed, axis=0)
y_true_animal = np.concatenate(y_true_animal, axis=0)

y_pred_probs_breed, y_pred_probs_animal = model.predict(test_ds)

y_pred_breed = np.argmax(y_pred_probs_breed, axis=1)
y_pred_animal = (y_pred_probs_animal > 0.5).astype(int)  # 0 = cat, 1 = dog

target_names = [breeds[i] for i in sorted(set(y_true_breed))]

print("📌 Classification Report for Breeds:")
print(classification_report(y_true_breed, y_pred_breed, labels=sorted(set(y_true_breed)), target_names=target_names))

print("\n📌 Classification Report for Animal Type (Cat vs Dog):")
print(classification_report(y_true_animal, y_pred_animal, labels=[0, 1], target_names=["Cat", "Dog"]))
