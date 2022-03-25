from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy

from data_preprocessing import load_data, train_generator
from unet import build_unet
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from deeplab import DeepLabV3Plus


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)
        # show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


if __name__ == '__main__':
    '''
    root_path = '/Users/debstutidas/PycharmProjects/opticDiskCup/data/Drishti-GS/train'
    images, masks = load_data(root_path)
    '''
    train_generator = train_generator()
    learning_rate = 0.1
    input_shape = (256, 256, 3)

    model = DeepLabV3Plus(input_shape)

    loss = CategoricalCrossentropy()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=["accuracy"],
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=5)



