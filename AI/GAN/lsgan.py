import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, LeakyReLU, Conv1DTranspose
from tensorflow.keras.optimizers import Adam

def build_generator(input_shape):
    x = Input(input_shape)

    y = Conv1DTranspose(512, 3, strides=1, padding="same")(x)
    y = LeakyReLU(0.2)(y)

    y = Conv1DTranspose(256, 3, strides=1, padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv1DTranspose(128, 3, strides=1, padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv1DTranspose(64, 3, strides=1, padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv1D(3, 3, padding="same", activation="tanh")(y)
    return Model(x,y)

def build_discriminator(input_shape):
    x = Input(input_shape)

    y = Conv1D(64, 3, strides=1, padding="same")(x)
    y = LeakyReLU(0.2)(y)

    y = Conv1D(128, 3, strides=1, padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv1D(256, 3, strides=1, padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv1D(512, 3, strides=1, padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = GlobalAveragePooling1D()(y)
    y = Dense(1)(y)
    return Model(x, y)


def build_train_step(generator, discriminator):
    d_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)
    g_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)

    @tf.function
    def train_step(real_image, noise):
        fake_image = generator(noise)

        pred_real = discriminator(real_image)
        pred_fake = discriminator(fake_image)

        LAMBA = tf.cast(0.0002, dtype=tf.float32)

        pred_real = tf.cast(pred_real, dtype=tf.float32)
        pred_fake = tf.cast(pred_fake, dtype=tf.float32)

        real_image = tf.cast(real_image, dtype=tf.float32)
        fake_image = tf.cast(fake_image, dtype=tf.float32)

        d_loss = tf.reduce_mean(tf.maximum(pred_real - pred_fake + LAMBA * tf.reduce_sum(tf.abs(real_image-fake_image)), 0.0))
        g_loss = tf.reduce_mean(pred_fake)

        d_gradients = tf.gradients(d_loss, discriminator.trainable_variables)
        g_gradients = tf.gradients(g_loss, generator.trainable_variables)

        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        return d_loss, g_loss

    return train_step
