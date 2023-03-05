from keras.layers import Input, Conv3D, Conv3DTranspose, BatchNormalization, Activation, LeakyReLU
from keras.models import Model

def build_generator():
    inputs = Input((1, 1, 1, 200))
    x = Conv3DTranspose(32*8, kernel_size=4, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(32*4, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(32*2, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(32, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(32 / 2, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3DTranspose(32 / 4, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    outputs = Conv3D(1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)
    return Model(inputs, outputs)


def build_discriminator():
    inputs = Input((32, 32, 32, 1))
    x = Conv3D(32, kernel_size=4, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv3D(32*2, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv3D(32*4, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv3D(32*8, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    outputs = Conv3D(1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)
    return Model(inputs, outputs)
