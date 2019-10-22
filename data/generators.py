import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight


class DataGenerator:

    def __init__(self, params):
        self.params = params
        self.data_root = self.params['data']['root_path']
        shape = self.params['model']['shape'].split(":")
        self.target_size = (int(shape[0]), int(shape[1]))

    def create_generators(self):
        train_gen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            fill_mode='nearest',
            horizontal_flip=True)

        validation_gen = ImageDataGenerator(rescale=1. / 255)

        test_gen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_gen.flow_from_directory(
            self.data_root + 'train/',
            target_size=self.target_size,
            color_mode='rgb',
            shuffle=True,
            batch_size=self.params['model']['train_generator_batch_size'],
            class_mode='categorical',
            # save_to_dir=data + 'prepared/',
            seed=1)

        validation_generator = validation_gen.flow_from_directory(
            self.data_root + 'validation/',
            target_size=self.target_size,
            color_mode='rgb',
            shuffle=True,
            batch_size=self.params['model']['validation_generator_batch_size'],
            class_mode='categorical',
            seed=1)

        test_generator = test_gen.flow_from_directory(
            self.data_root + 'test/',
            target_size=self.target_size,
            color_mode='rgb',
            shuffle=False,
            batch_size=1,
            class_mode='categorical',
            seed=1)

        train_generator_class_weight = class_weight \
            .compute_class_weight(
                'balanced',
                np.unique(train_generator.classes),
                train_generator.classes
            )

        return train_generator, validation_generator, test_generator, train_generator_class_weight
