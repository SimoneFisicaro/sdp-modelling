import argparse
import datetime
import logging
import time
from os import path

import mlflow
import mlflow.keras
import yaml

from data.generators import DataGenerator
from model.modelling import Modelling


def run(params):
    logger = create_logger(params['logger']['log_folder'], params['logger']['file_prefix'])
    prepare_mlflow(params['mlflow'])

    with mlflow.start_run(nested=True):
        logger.info("Starting training.")

        mlflow.keras.autolog()
        mlflow.log_params(params['model'])
        logger.info("Input parameters: \n{0}".format(params['model']))

        generator = DataGenerator(params)
        model = Modelling(params, generator)
        model.run()


def create_logger(log_folder, file_prefix):
    logger = logging.getLogger()

    file_name = file_prefix + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.log'

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path.join(log_folder, file_name))
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def load_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='parameters_file', metavar='f', help='Path to parameters file.')
    args = parser.parse_args()

    with open(args.parameters_file, 'r') as stream:
        return yaml.safe_load(stream)


def prepare_mlflow(params):
    mlflow.set_tracking_uri(params['tracking_uri'])
    mlflow.set_experiment(params['experiment'])


if __name__ == "__main__":
    params = load_parameters()

    run(params)
