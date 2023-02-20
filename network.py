from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU,Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def AutoEncoder(cfg):

    input_img = Input(shape=(cfg.patch_size, cfg.patch_size, cfg.input_channel))

    h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='valid')(input_img)  # 32
    h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 32
    if cfg.patch_size == 256:
        h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 32
    h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 32
    h = Conv2D(cfg.flc * 2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 64
    h = Conv2D(cfg.flc * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 64
    h = Conv2D(cfg.flc * 4, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 128
    h = Conv2D(cfg.flc * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 64
    h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 32
    encoded = Conv2D(cfg.z_dim, (8, 8), strides=1, activation='linear', padding='valid')(h)

    h = Conv2DTranspose(cfg.flc, (8, 8), strides=1, activation=LeakyReLU(alpha=0.2), padding='valid')(encoded)  # 32
    h = Conv2D(cfg.flc * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 64
    h = Conv2D(cfg.flc * 4, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 128
    h = Conv2DTranspose(cfg.flc * 2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 64
    h = Conv2D(cfg.flc * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 64
    h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 32
    h = Conv2D(cfg.flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 32
    h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 32
    if cfg.patch_size == 256:
        h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)  # 32

    decoded = Conv2DTranspose(cfg.input_channel, (4, 4), strides=2, activation='sigmoid', padding='same')(h)

    return Model(input_img, decoded)
