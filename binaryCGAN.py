import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# set up files and hyperparams
IMG_SIZE = 128
BATCH_SIZE = 64
NOISE_DIM = 128
CSV_PATH = 'train_combined/train_combined.csv'
IMG_DIR = 'train_combined/train_images'

# load in data and label each image as bird/butterfly
df = pd.read_csv(CSV_PATH)
target_categories = ['bird', 'butterfly']
df = df[df['category'].isin(target_categories)].copy()
classes = sorted(df['category'].unique())
class_map = {name: i for i, name in enumerate(classes)}
num_classes = len(classes)

# organizing all the file paths
img_paths = [os.path.join(IMG_DIR, f) for f in df['filename']]
labels = [class_map[l] for l in df['category']]

# augment data by flipping, rotating, and zooming
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def load_img(path, label):
    '''loads image from file path and normalizes pixels'''
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img - 127.5) / 127.5
    return img, label

# set up dataset for training by loading images, doing data augmentation, and shuffling
dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_generator(latent_dim, num_classes):
    '''generator model architecture'''
    # embedding set up
    label_in = layers.Input(shape=(1,), name='gen_label_in')
    l = layers.Embedding(num_classes, 50)(label_in)
    l = layers.Dense(8 * 8 * 1)(l)
    l = layers.Reshape((8, 8, 1))(l) 

    # noise set up
    noise_in = layers.Input(shape=(latent_dim,), name='gen_noise_in')
    n = layers.Dense(8 * 8 * 512)(noise_in)
    n = layers.Reshape((8, 8, 512))(n)

    # combine embedding and noise
    x = layers.Concatenate()([n, l])

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(256, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    out = layers.Conv2D(3, (5,5), activation='tanh', padding='same')(x)
    return models.Model([noise_in, label_in], out, name="Generator")

def build_discriminator(img_size, num_classes):
    '''disciminator model architecture'''
    img_in = layers.Input(shape=(img_size, img_size, 3), name='disc_img_in')
    label_in = layers.Input(shape=(1,), name='disc_label_in')

    # set up embedding
    l = layers.Embedding(num_classes, 50)(label_in)
    l = layers.Dense(img_size * img_size * 1)(l)
    l = layers.Reshape((img_size, img_size, 1))(l)

    # combine image and embedding
    merge = layers.Concatenate()([img_in, l])

    x = layers.Conv2D(64, (5,5), strides=(2,2), padding='same')(merge)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (5,5), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model([img_in, label_in], out, name="Discriminator")

class CGAN(tf.keras.Model):
    '''class to set up the general CGAN architecture'''
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
        '''performs one step of training'''
        real_images, labels = data
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_images = self.generator([random_latent_vectors, labels])

        real_labels = tf.ones((batch_size, 1))
        real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
        
        fake_labels = tf.zeros((batch_size, 1))
        
        with tf.GradientTape() as tape:
            real_predictions = self.discriminator([real_images, labels])
            fake_predictions = self.discriminator([generated_images, labels])
            
            d_loss_real = self.loss_fn(real_labels, real_predictions)
            d_loss_fake = self.loss_fn(fake_labels, fake_predictions)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

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
    '''this class is used to save images of generator output at the end of each epoch'''
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
        
        filename = f"training_progress/epoch_{epoch+1}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

# set up model
gen = build_generator(NOISE_DIM, num_classes)
disc = build_discriminator(IMG_SIZE, num_classes)
cgan = CGAN(gen, disc, NOISE_DIM)
cgan.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# monitor for progress image output
monitor = GANMonitor(num_img=16, latent_dim=NOISE_DIM)

# train model
cgan.fit(dataset, epochs=1000, callbacks=[monitor])


# save weights when done training
cgan.generator.save_weights('binary_cgan_generator.weights.h5')
cgan.discriminator.save_weights('binary_cgan_discriminator.weights.h5')