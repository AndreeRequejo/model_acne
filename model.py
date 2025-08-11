import os
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

DATASET_DIR = "J:/1. SEMINARIO DE TESIS II/model_acne/ACNE04/all_1024"

filepaths = []
labels = []
label_map = {"levle0": "0", "levle1": "1", "levle2": "2", "levle3": "3"}
pattern = re.compile(r"levle[0-3]")
for fname in os.listdir(DATASET_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        match = pattern.search(fname)
        if match:
            label = label_map.get(match.group(), None)
            if label is not None:
                filepaths.append(os.path.join(DATASET_DIR, fname))
                labels.append(label)
df = pd.DataFrame({"filepath": filepaths, "label": labels})

# === Balanceo de clases ===
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label']),
    y=df['label']
)
class_weight_dict = {int(k): v for k, v in zip(np.unique(df['label']), class_weights)}

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# === 5. RESNET50 TRANSFER LEARNING ===
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # congelar capas base

x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# === 6. CALLBACKS ===
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_resnet50_acne04.h5", save_best_only=True, monitor="val_loss")
]

# === 7. ENTRENAMIENTO FASE 1 ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# === 8. AFINAR (UNFREEZE) FASE 2 ===
# Descongelar solo las últimas 30 capas para fine-tuning progresivo
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# === 9. GUARDAR MODELO FINAL ===
model.save("resnet50_acne04_final.h5")

# === SUGERENCIA: Generación sintética ===
# Si necesitas más imágenes, puedes usar librerías como imgaug o albumentations para generar datos sintéticos adicionales.
# Ejemplo:
# from imgaug import augmenters as iaa
# seq = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     iaa.Affine(rotate=(-20, 20)),
#     iaa.GaussianBlur(sigma=(0, 1.0)),
# ])