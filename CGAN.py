import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

IMG_SIZE = 128
BATCH_SIZE = 32
NOISE_DIM = 128
CSV_PATH = 'train_combined/train_combined.csv'
IMG_DIR = 'train_combined/train_images'

df = pd.read_csv(CSV_PATH)

classes = df['label'].unique()
class_map = {name: i for i, name in enumerate(classes)}
num_classes = len(classes)

img_paths = [os.path.join(IMG_DIR, f) for f in df['filename']]
labels = [class_map[l] for l in df['label']]

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img - 127.5) / 127.5
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_generator(latent_dim, num_classes):
    label_in = layers.Input(shape=(1,), name='gen_label_in')
    l = layers.Embedding(num_classes, 50)(label_in)
    l = layers.Dense(8 * 8 * 1)(l)
    l = layers.Reshape((8, 8, 1))(l) 

    noise_in = layers.Input(shape=(latent_dim,), name='gen_noise_in')
    n = layers.Dense(8 * 8 * 128)(noise_in)
    n = layers.Reshape((8, 8, 128))(n)

    merge = layers.Concatenate()([n, l])

    x = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)

    out = layers.Conv2D(3, (7,7), activation='tanh', padding='same')(x)

    return models.Model([noise_in, label_in], out, name="Generator")

def build_discriminator(img_size, num_classes):
    img_in = layers.Input(shape=(img_size, img_size, 3), name='disc_img_in')
    label_in = layers.Input(shape=(1,), name='disc_label_in')

    l = layers.Embedding(num_classes, 50)(label_in)
    l = layers.Dense(img_size * img_size * 1)(l)
    l = layers.Reshape((img_size, img_size, 1))(l)

    merge = layers.Concatenate()([img_in, l])

    x = layers.Conv2D(64, (4,4), strides=(2,2), padding='same')(merge)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, (4,4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, (4,4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return models.Model([img_in, label_in], out, name="Discriminator")

class CGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(CGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(CGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator([random_latent_vectors, labels])

        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_labels = tf.concat([labels, labels], axis=0)
        
        labels_discriminator = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)

        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, combined_labels])
            d_loss = self.loss_fn(labels_discriminator, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator([self.generator([random_latent_vectors, labels]), labels])
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=16, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
        self.random_labels = tf.random.uniform([num_img, 1], minval=0, maxval=num_classes, dtype=tf.int32)

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator([self.random_latent_vectors, self.random_labels])
        generated_images = (generated_images * 127.5) + 127.5

        fig = plt.figure(figsize=(4, 4))
        for i in range(self.num_img):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_images[i].numpy().astype("uint8"))
            plt.axis('off')
        
        filename = f"training_progress/epoch_{epoch+1}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

gen = build_generator(NOISE_DIM, num_classes)
disc = build_discriminator(IMG_SIZE, num_classes)
cgan = CGAN(gen, disc, NOISE_DIM)

cgan.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

monitor = GANMonitor(num_img=16, latent_dim=NOISE_DIM)

cgan.fit(dataset, epochs=50, callbacks=[monitor])

cgan.generator.save_weights('cgan_generator.weights.h5')
cgan.discriminator.save_weights('cgan_discriminator.weights.h5')