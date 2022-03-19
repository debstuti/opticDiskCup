import os
import glob
import cv2
import numpy as np
import tensorflow as tf


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


def tf_dataset(images, masks, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset
