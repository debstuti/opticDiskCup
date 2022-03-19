from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model


def conv_block(input, n_filter):
    conv = Conv2D(n_filter, kernel_size=(3, 3), activation='relu', padding='same')(input)
    conv = Conv2D(n_filter, kernel_size=(3, 3), activation='relu', padding='same')(conv)
    return conv


def down_sample(input, n_filter):
    conv = conv_block(input, n_filter)
    next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, next_layer


def up_sample(inputs, skip_layer_input, n_filter):
    conv = Conv2DTranspose(n_filter, kernel_size=(2, 2), strides=2, padding='same')(inputs) ##todo : understand
    merged = Concatenate()([conv, skip_layer_input])
    conv = conv_block(merged, n_filter)
    return conv


def build_unet(input_shape):
    inputs = Input(input_shape)

    """ DownSample"""
    d1, a1 = down_sample(inputs, 64)
    d2, a2 = down_sample(a1, 128)
    d3, a3 = down_sample(a2, 256)
    d4, a4 = down_sample(a3, 512)

    """ Bridge """
    b1 = conv_block(a4, 1024)

    """ UpSample"""
    u1 = up_sample(b1, d4, 512)
    u2 = up_sample(u1, d3, 256)
    u3 = up_sample(u2, d2, 128)
    u4 = up_sample(u3, d1, 64)

    """ Outputs """
    outputs = Conv2D(1, 1, activation='sigmoid')(u4)

    model = Model(inputs, outputs)
    return model
