import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(root_path):
    images = sorted(glob.glob(os.path.join(root_path, 'image/img/*')))
    mask = sorted(glob.glob(os.path.join(root_path, 'mask/img/*')))
    return images, mask


def read_image(image_path):
    img = cv2.imread(image_path)
    if img:
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = img.astype(np.float32)
        # cv2.imshow('image0', img)
        # cv2.waitKey(0)  # wait for a keyboard input
        # cv2.destroyAllWindows()
    return img


def read_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask:
        mask = cv2.resize(mask, (256, 256))
        mask = mask / 255.0
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, -1)
    return mask


def preprocess(image, mask):
    def f(img, mask):
        img = read_image(img)
        mask = read_mask(mask)
        return img, mask

    image, mask = tf.numpy_function(f, [image, mask], [tf.float32, tf.float32])
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 1])
    return image, mask

def train_generator():
    # we create two instances with the same arguments
    data_gen_args = dict(vertical_flip=True,
                         horizontal_flip=True,
                         rescale=1./255)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_generator = image_datagen.flow_from_directory(
        '/Users/debstutidas/PycharmProjects/opticDiskCup/data/Drishti-GS/train/image',
        class_mode=None,
        classes=None,
        seed=seed,
        batch_size=16)
    mask_generator = mask_datagen.flow_from_directory(
        '/Users/debstutidas/PycharmProjects/opticDiskCup/data/Drishti-GS/train/mask',
        class_mode=None,
        seed=seed,
        batch_size=16)
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

