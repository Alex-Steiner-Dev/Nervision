import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm
import trimesh
import glob
import os

DATA_DIR = "../Data/*"

def parse_dataset(num_points=2048):
    train_points = []
    train_labels = []
    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        print("Processing class: {}".format(os.path.basename(folder)))
    
        train_files = glob.glob(folder + "/train/*")
        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

    return (
        np.array(train_points),
        np.array(train_labels),
    )

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, train_labels = parse_dataset()

def augment(points, label):
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    points = tf.random.shuffle(points)

    return points, label

dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))

discriminator = keras.Sequential(
    [
        keras.Input(shape=(2048, 2048, 3)),
        layers.Conv2D(2048, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(4096, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(4096, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()

latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 4096),
        layers.Reshape((8, 8, 4096)),
        layers.Conv2DTranspose(4096, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()

opt_gen = keras.optimizers.Adam(1e-4)
opt_disc = keras.optimizers.Adam(1e-4)
loss_fn = keras.losses.BinaryCrossentropy()

for epoch in range(10):
    for idx, (real) in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        with tf.GradientTape() as gen_tape:
            random_latent_vectors = tf.random.normal(shape = (batch_size, latent_dim))
            fake = generator(random_latent_vectors)

        if idx % 100 == 0:
            print(fake[0])

        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss_fn(tf.ones((batch_size, 1)), discriminator(real))
            loss_disc_fake = loss_fn(tf.zeros((batch_size, 1)), discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake)/2

        grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(
            zip(grads, discriminator.trainable_weights)
        )

        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size, 1), output)

        grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(zip(grads, generator.trainable_weights))
