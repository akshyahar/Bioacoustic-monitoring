import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset/features/cnn_dataset"
MODEL_DIR = "models"

IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 8
EPOCHS = 50

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading CNN dataset")
print("=" * 60)

train_dir = os.path.join(
    DATASET_DIR,
    "train"
)

validation_dir = os.path.join(
    DATASET_DIR,
    "validation"
)

test_dir = os.path.join(
    DATASET_DIR,
    "test"
)

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        i,
        "->",
        name
    )

print(
    "\nNumber of classes:",
    num_classes
)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.10
    ),

])

# ============================================================
# NORMALIZATION
# ============================================================

normalization_layer = layers.Rescaling(
    1.0 / 255
)

# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    ),

    data_augmentation,

    normalization_layer,

    # First convolution block

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution block

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Third convolution block

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Fourth convolution block

    layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Classification

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        num_classes,
        activation="softmax"
    )

])

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nCNN Model")
print("=" * 60)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "forest_health_cnn.keras"
)

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    checkpoint_path,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"

)

# ============================================================
# TRAIN
# ============================================================

print("\nStarting CNN training")
print("=" * 60)

history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]

)

# ============================================================
# TEST
# ============================================================

print("\nEvaluating model on test dataset")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(
    test_ds
)


print(
    "\nTest Loss:",
    round(
        test_loss,
        4
    )
)


print(
    "Test Accuracy:",
    round(
        test_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_file = os.path.join(
    MODEL_DIR,
    "class_names.json"
)


with open(
    class_file,
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = os.path.join(
    MODEL_DIR,
    "training_history.json"
)


history_data = {
    key: [
        float(value)
        for value in values
    ]
    for key, values
    in history.history.items()
}


with open(
    history_file,
    "w"
) as f:

    json.dump(
        history_data,
        f,
        indent=4
    )

# ============================================================
# COMPLETED
# ============================================================

print("\nModel saved to:")
print(checkpoint_path)

print("\nClass names saved to:")
print(class_file)

print("\nTraining history saved to:")
print(history_file)

print("\nCNN training completed!")