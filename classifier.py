import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "train_combined/train_images"
CSV_PATH = "train_combined/train_combined.csv"
IMG_SIZE, BATCH_SIZE = 128, 64

# set up data
df = pd.read_csv(CSV_PATH, dtype=str)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_dataframe(
    df, directory=DATA_DIR, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, subset='training'
)

val_gen = datagen.flow_from_dataframe(
    df, directory=DATA_DIR, x_col='filename', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, subset='validation', shuffle=False
)

# fine tuning on top of EfficientNetB0 base
base = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(train_gen, validation_data=val_gen, epochs=20)

# show 9 random classsifications
x_val, y_val = next(val_gen)
preds = model.predict(x_val, verbose=0)
labels = {v: k for k, v in train_gen.class_indices.items()}

plt.figure(figsize=(12, 12))
indices = np.random.choice(len(x_val), size=min(9, len(x_val)), replace=False)

for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_val[idx].astype("uint8")) 
    
    true = labels[np.argmax(y_val[idx])]
    pred = labels[np.argmax(preds[idx])]
    plt.title(f"T: {true}\nP: {pred}", color='green' if true == pred else 'red')
    plt.axis('off')
plt.show()

model.save("efficientnet_classifier.h5")