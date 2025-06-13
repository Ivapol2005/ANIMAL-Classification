from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from breeds_list import breeds
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy

base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

input_layer = base_model.input
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output_breed = layers.Dense(len(breeds), activation='softmax', name="breed_output")(x)
output_animal = layers.Dense(1, activation='sigmoid', name="animal_output")(x)
output_bbox = layers.Dense(4, activation='linear', name="bbox_output")(x)

model = models.Model(inputs=input_layer, outputs={
    "breed_output": output_breed,
    "animal_output": output_animal,
    "bbox_output": output_bbox
})

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

model.compile(
    optimizer='adam',
    loss={
        "breed_output": CategoricalCrossentropy(label_smoothing=0.1),
        "animal_output": BinaryCrossentropy(),
        "bbox_output": MeanSquaredError()
    },
    metrics={
        "breed_output": "accuracy",
        "animal_output": "accuracy",
        "bbox_output": "mse"
    }
)

model.summary()