import tensorflow as tf
import matplotlib.pyplot as plt

from evaluation.model_evaluator import ModelEvaluator
from experiments.output_path_builder import OutputPathBuilder
from cwae_model import CwaeModel
from dataset_loaders import Dataset
from experiments.canvas_fillers import RgbCanvasFiller



class TrainingProcess:

    def __init__(self, dataset: Dataset, model: CwaeModel, model_evaluator: ModelEvaluator,
                 path_builder: OutputPathBuilder, optimizer: tf.train.Optimizer, canvas_filler: RgbCanvasFiller):
        self.__model = model
        self.__model_evaluator = model_evaluator
        self.__dataset = dataset
        self.__path_builder = path_builder
        self.__optimizer = optimizer
        self.__canvas_filler = canvas_filler

    def get_model(self) -> CwaeModel:
        return self.__model

    def start(self, tf_session: tf.Session):
        print('Starting training process')
        self.__tf_session = tf_session

        print('Building cost tensor')
        self.__tensor_cost = self.__model.build_cost_tensor()
        print('Building training ops')
        self.__tensor_train_ops = [self.__optimizer.minimize(self.__tensor_cost)]

        print('Initializing global vars')
        self.__tf_session.run(tf.global_variables_initializer())
        print('Training setup complete')

    def next_epoch(self, epoch_number: int):
        try:
            while True:
                self.__tf_session.run(self.__tensor_train_ops)
        except tf.errors.OutOfRangeError:
            pass
        self.__path_builder.set_prefix(epoch_number)

    def end(self):
        print('Ending training process')
        with open(self.__path_builder.get_path('report.json'), 'w') as json_file:
            json_file.write(self.__model_evaluator.get_json())
        print('Results available in directory:', self.__path_builder.get_base_dir())

    def evaluate(self, epoch_number: int):
        res, z = self.__model_evaluator.evaluate(self.__tf_session,
                                                 self.__dataset.validation_images[0:], epoch_number)
        print(res)
        return res, z

    def save_image(self, name, images, size=(1, 1)):
        path = self.__path_builder.get_path(name)
        if size == (1, 1):
            canvas = self.__canvas_filler.build_canvas([images], (1, 1), self.__dataset.x_dim)
        else:
            canvas = self.__canvas_filler.build_canvas(images, size, self.__dataset.x_dim)

        if self.__canvas_filler.cmap is None:
            plt.imsave(path, canvas)
        else:
            plt.imsave(path, canvas, cmap=self.__canvas_filler.cmap)
