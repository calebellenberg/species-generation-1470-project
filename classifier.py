import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_ROOT = "./Birds"
IMG_SIZE, BATCH_SIZE = 224, 32

train_df = pd.read_csv(f'{DATA_ROOT}/train.csv', dtype=str).dropna()
test_df = pd.read_csv(f'{DATA_ROOT}/test.csv', dtype=str)

def fix_filename(fn):
    return fn if str(fn).endswith('.jpg') else f"{fn}.jpg"

train_df['filename'] = train_df['img_id'].apply(fix_filename)
test_df['filename'] = test_df['img_id'].apply(fix_filename)

aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = aug.flow_from_dataframe(
    train_df,
    directory=f'{DATA_ROOT}/train',
    x_col='filename',
    y_col='target',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

val_gen = aug.flow_from_dataframe(
    train_df,
    directory=f'{DATA_ROOT}/train',
    x_col='filename',
    y_col='target',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=False
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
    test_df,
    directory=f'{DATA_ROOT}/test',
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

base = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights=None)
weights_path = tf.keras.utils.get_file(
    'efficientnetb0_notop.h5',
    'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
    cache_subdir='models'
)
base.load_weights(weights_path)
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_gen, validation_data=val_gen, epochs=5)

test_gen.reset()
preds = model.predict(test_gen, verbose=1)
pred_classes = np.argmax(preds, axis=1)

idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
pred_labels = [idx_to_label[i] for i in pred_classes]

val_gen.reset()
x_val, y_val = next(val_gen)
preds_val = model.predict(x_val, verbose=0)

plt.figure(figsize=(16, 16))
for i in range(min(16, len(x_val))):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_val[i].astype("uint8"))
    
    true_label = idx_to_label[np.argmax(y_val[i])]
    pred_label = idx_to_label[np.argmax(preds_val[i])]
    
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()