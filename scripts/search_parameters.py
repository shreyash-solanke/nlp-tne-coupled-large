
import os

from allennlp.common.params import Params
from allennlp.commands.train import train_model

from tne.modeling.models.tne_coupled import TNECoupledModel
from tne.modeling.dataset_readers.tne_reader import TNEReader
from tne.modeling.predictors.tne_predictor import TNEPredictor


def search_params():
    # this function is used to search different parameters over a model

    serialization_dir = 'models/coupled_spanbert_large'
    include_package = ['tne']
    recover = False
    file_friendly_logging = False

    base_parameter_filename = 'tne/modeling/configs/coupled_large.jsonnet'
    params = Params.from_file(base_parameter_filename)

    parameters_sets = [{'length': 1, 'width': 6}, {'length': 2, 'width': 6}, {'length': 1, 'width': 8},
                       {'length': 2, 'width': 8}]

    for parameters_set in parameters_sets:
        params.update(parameters_set)
        train_model(params, serialization_dir=serialization_dir, include_package=include_package,
                    recover=recover, file_friendly_logging=file_friendly_logging)


if __name__ == '__main__':
    os.chdir('/scratch/ssolunke/cs678-proj/TNE/')
    search_params()