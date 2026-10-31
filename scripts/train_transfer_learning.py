import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


DATASET_DIR = "dataset/features/cnn_dataset"
MODEL_DIR = "models"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 8
EPOCHS = 30

os.makedirs(MODEL_DIR, exist_ok=True)


print("Loading dataset")
print("=" * 60)


train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.05),

    layers.RandomZoom(0.1),

])


# ============================================================
# PRETRAINED MOBILE NET
# ============================================================

base_model = MobileNetV2(

    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained layers initially
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    )
)


x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\nModel")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "forest_health_mobilenet.keras"
)


checkpoint = ModelCheckpoint(

    model_path,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting transfer-learning training")
print("=" * 60)


history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating on test dataset")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(
    test_ds
)


print(
    "\nTest Loss:",
    round(test_loss, 4)
)


print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_file = os.path.join(
    MODEL_DIR,
    "mobilenet_class_names.json"
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


print("\nModel saved to:")
print(model_path)


print("\nClass names saved to:")
print(class_file)


print("\nTransfer learning completed!")