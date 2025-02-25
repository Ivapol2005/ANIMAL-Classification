from sklearn.metrics import classification_report
# from breeds_list import breeds, cat_breeds, dog_breeds

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

