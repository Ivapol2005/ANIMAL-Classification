from create_dataset import train_ds, test_ds, dataset_train, dataset_test
from model_settings import model, early_stop, reduce_lr

for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label keys:", labels.keys())
    print("breed_output shape:", labels["breed_output"].shape)
    print("animal_output shape:", labels["animal_output"].shape)


BATCH_SIZE = 32

train_steps_per_epoch = len(dataset_train) // BATCH_SIZE
if len(dataset_train) % BATCH_SIZE != 0:
    train_steps_per_epoch += 1

validation_steps = len(dataset_test) // BATCH_SIZE
if len(dataset_test) % BATCH_SIZE != 0:
    validation_steps += 1


model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=[early_stop, reduce_lr],
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=validation_steps
)

model.save('../cat_dog_classifier.h5')
