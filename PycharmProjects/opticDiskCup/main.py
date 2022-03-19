from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = 128
data_generator = ImageDataGenerator()
image_generator = data_generator.flow_from_directory(
                                        directory="/Users/debstutidas/PycharmProjects/opticDiskCup/data/Drishti-GS/train/image",
                                        target_size=(IMG_SIZE, IMG_SIZE),
                                        batch_size=16,
                                        class_mode=None,
                                        classes=None
                                        )

mask_generator = data_generator.flow_from_directory(
    directory="/Users/debstutidas/PycharmProjects/opticDiskCup/data/Drishti-GS/train/mask",
    class_mode=None,
    classes=None,
    batch_size=1,
    )

train_generator = zip(image_generator, mask_generator)