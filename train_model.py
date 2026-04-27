# train_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
import os

print("TensorFlow version:", tf.__version__)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_PATH = r'C:\Users\Mohith\Downloads\NOTES\PROGAMMING\waste classifer\TRAIN.1'
MODEL_SAVE_PATH = r'C:\Users\Mohith\Downloads\NOTES\PROGAMMING\waste classifer\models'

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print(f"\n{'='*60}")
print("TRAINING WASTE CLASSIFIER MODEL")
print(f"{'='*60}")

# Data preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print(f"\nTraining images: {train_generator.samples}")
print(f"Validation images: {validation_generator.samples}")

# Build model
def create_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

model = create_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, 'waste_classifier.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# Train
print("\nStarting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save
model.save(os.path.join(MODEL_SAVE_PATH, 'waste_classifier.h5'))

class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open(os.path.join(MODEL_SAVE_PATH, 'class_labels.json'), 'w') as f:
    json.dump(class_labels, f)

print("\n✓ Training complete!")
print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
