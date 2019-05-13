import os

from experiments.training_process import TrainingProcess
from experiments.output_path_builder import OutputPathBuilder
from evaluation.model_evaluator import ModelEvaluator


class TrainingProcessFactory:

    def __init__(self, dataset, base_dir: str, model_factory, architecture_provider,
                 optimizer_factory):
        self.__dataset = dataset
        self.__base_dir = base_dir
        self.__model_factory = model_factory
        self.__architecture_provider = architecture_provider
        self.__optimizer_factory = optimizer_factory

    def create(self, latent_size: int, experiment_name: str) -> TrainingProcess:
        if (latent_size < 1):
            raise ValueError(f'Invalid latent dimension: {latent_size}')

        model = self.__model_factory.create(self.__dataset, latent_size)
        model.build_architecture(self.__architecture_provider.encoder_builder,
                                 self.__architecture_provider.decoder_builder)

        experiment_path = os.path.join(self.__base_dir, experiment_name)
        output_path_builder = OutputPathBuilder(experiment_path)
        optimizer = self.__optimizer_factory()

        model_evaluator = ModelEvaluator(model)
        model_evaluator.build()

        return TrainingProcess(self.__dataset, model, model_evaluator,
                               output_path_builder, optimizer, self.__dataset.canvas_filler)
