from tensorflow.keras.losses import SparseCategoricalCrossentropy

from data_preprocessing import load_data, tf_dataset
from unet import build_unet
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from deeplab import Deeplabv3


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)
        # show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


if __name__ == '__main__':
    root_path = '/Users/debstutidas/PycharmProjects/opticDiskCup/data/Drishti-GS/train'
    images, masks = load_data(root_path)
    dataset = tf_dataset(images, masks, 2)

    learning_rate = 0.1

    model = Deeplabv3(classes=2)
    model.compile(loss='binary_crossentropy')
    print(model.summary())
    ''' 
    model = build_unet((256, 256, 3))
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate))
    print(model.summary())

    csv_path = 'output/out.csv'

    batch_size = 32
    n_epochs = 1

    train_steps_per_batch = len(images) // batch_size

    callbacks = [
        # ModelCheckpoint(model_path, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=5),
        # CSVLogger(csv_path),
        TensorBoard()
    ]

    model_history = model.fit(dataset, epochs=n_epochs,
                              steps_per_epoch=train_steps_per_batch,
                              validation_data=dataset,
                              validation_steps=train_steps_per_batch,
                              callbacks=callbacks)

    
    '''
