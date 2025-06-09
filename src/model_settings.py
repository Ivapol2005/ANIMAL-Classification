from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from code.breeds_list import *

# Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(breeds), activation='softmax')  # Output layer for classification
])


base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

input_layer = base_model.input
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output_breed = layers.Dense(len(breeds), activation='softmax', name="breed_output")(x)
output_animal = layers.Dense(1, activation='sigmoid', name="animal_output")(x)

model = models.Model(inputs=input_layer, outputs=[output_breed, output_animal])

base_model.trainable = False


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

model.compile(optimizer='adam',
              loss={'breed_output': 'sparse_categorical_crossentropy',
                    'animal_output': 'binary_crossentropy'},
              metrics={'breed_output': 'accuracy', 'animal_output': 'accuracy'})
