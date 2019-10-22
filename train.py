import argparse

import mlflow
import mlflow.keras
import yaml

from data.generators import DataGenerator
from model.modelling import Modelling


def run(params):
    prepare_mlflow(params['mlflow'])

    with mlflow.start_run(nested=True):

        mlflow.keras.autolog()
        mlflow.log_params(params['model'])

        generator = DataGenerator(params)
        model = Modelling(params, generator)
        model.run()


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
