
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.metrics import f1_score

# Dataset configuration
DATASET_PATH = 'oil_spill_dataset'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Create base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom head with batch normalization
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile with AUC metric
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', AUC(name='auc')])

# Callbacks with longer patience
callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
    ModelCheckpoint('oil_spill_model.h5', save_best_only=True, monitor='val_auc', mode='max'),
    ReduceLROnPlateau(factor=0.1, patience=7, monitor='val_auc', mode='max')
]

# Initial training
print("Initial training...")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks
)

# Fine-tuning
print("Fine-tuning the model...")
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy', AUC(name='auc')])

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks
)

# Save final model
model.save('oil_spill_model.keras')

# Calculate and save optimal threshold
val_pred = model.predict(val_generator)
val_labels = val_generator.labels

# Find threshold that maximizes F1-score
thresholds = np.arange(0.1, 1.0, 0.05)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    pred_labels = (val_pred >= threshold).astype(int)
    try:
        f1 = f1_score(val_labels, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    except ZeroDivisionError:
        continue

print(f"Optimal threshold: {best_threshold:.2f} (F1-score: {best_f1:.2f})")

# Save threshold to a file
with open('model_threshold.txt', 'w') as f:
    f.write(str(best_threshold))

