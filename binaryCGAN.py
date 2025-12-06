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

target_categories = ['bird', 'butterfly']
df = df[df['category'].isin(target_categories)].copy()

classes = sorted(df['category'].unique())
class_map = {name: i for i, name in enumerate(classes)}
num_classes = len(classes)

print(f"Training on {len(df)} images with {num_classes} categories: {classes}")

img_paths = [os.path.join(IMG_DIR, f) for f in df['filename']]
labels = [class_map[l] for l in df['category']]

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
    n = layers.Dense(8 * 8 * 512)(noise_in)
    n = layers.Reshape((8, 8, 512))(n)

    x = layers.Concatenate()([n, l])

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
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
    
    x = layers.Conv2D(512, (4,4), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)
    return models.Model([img_in, label_in], out, name="Discriminator")

class CGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim, gp_weight=10.0):
        super(CGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

    def compile(self, g_optimizer, d_optimizer):
        super(CGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images, labels):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, labels], training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]
        
        for i in range(5):
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
                fake_images = self.generator([random_latent_vectors, labels], training=True)

                real_logits = self.discriminator([real_images, labels], training=True)
                fake_logits = self.discriminator([fake_images, labels], training=True)

                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images, labels)
                d_loss = d_cost + (gp * self.gp_weight)

            d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            fake_images = self.generator([random_latent_vectors, labels], training=True)
            gen_logits = self.discriminator([fake_images, labels], training=True)
            
            g_loss = -tf.reduce_mean(gen_logits)

        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=16, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
        self.random_labels = tf.constant([i % 2 for i in range(num_img)], dtype=tf.int32)
        self.random_labels = tf.reshape(self.random_labels, (num_img, 1))

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator([self.random_latent_vectors, self.random_labels])
        generated_images = (generated_images * 127.5) + 127.5

        fig = plt.figure(figsize=(10, 10))
        for i in range(self.num_img):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_images[i].numpy().astype("uint8"))
            label_name = classes[int(self.random_labels[i])]
            plt.title(label_name)
            plt.axis('off')
        
        filename = f"training_progress_binary/epoch_{epoch+1}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

gen = build_generator(NOISE_DIM, num_classes)
disc = build_discriminator(IMG_SIZE, num_classes)

cgan = CGAN(gen, disc, NOISE_DIM)

cgan.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
)

monitor = GANMonitor(num_img=16, latent_dim=NOISE_DIM)

cgan.fit(dataset, epochs=50, callbacks=[monitor])

cgan.generator.save_weights('binary_cgan_generator.weights.h5')
cgan.discriminator.save_weights('binary_cgan_discriminator.weights.h5')