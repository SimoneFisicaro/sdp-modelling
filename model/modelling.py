from math import ceil

import mlflow
from keras import layers
from keras import optimizers
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.models import Model
from mlflow import keras


class Modelling:

    def __init__(self, params, generator):
        self.params = params
        self.generator = generator

    def run(self):
        base_model = self.load_base_model()
        model = self.build_model(base_model)

        (train_generator, validation_generator, test_generator, train_generator_class_weight) = \
            self.generator.create_generators()

        result = self.train(
            model,
            base_model,
            train_generator,
            validation_generator,
            train_generator_class_weight
        )

        if self.params['model']['save_model']:
            mlflow.keras.log_model(model, "models")

    def train(self, model, base_model, train_generator, validation_generator, train_generator_class_weight):
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(),
            loss=categorical_crossentropy,
            metrics=[categorical_accuracy])

        return model.fit_generator(
            train_generator,
            steps_per_epoch=ceil(len(train_generator) / self.params['model']['batch_size']),
            epochs=self.params['model']['epochs'],
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_generator) / self.params['model']['batch_size']),
            class_weight=train_generator_class_weight,
            workers=self.params['model']['workers'],
            verbose=0)

    def load_base_model(self):
        model = self.params['model']['base_network']
        shape = self.params['model']['shape'].split(":")
        input_shape = (int(shape[0]), int(shape[1]), int(shape[2]))

        if model == 'VGG16':
            return VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model == 'MobileNetV2':
            return MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model == 'InceptionResNetV2':
            return InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    def build_model(self, base_model):
        x = base_model.output
        x = layers.Flatten()(x)
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        predictions = layers.Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
