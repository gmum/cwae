import numpy as np
import tensorflow as tf
import json
from evaluation.mmd_evaluator import MMDEvaluator
from evaluation.cw_evaluator import CWEvaluator
from evaluation.mardia_evaluator import MardiaEvaluator
from evaluation.sw_evaluator import SWEvaluator
from rec_errors import squared_euclidean_norm_reconstruction_error
from cwae_model import CwaeModel


class ModelEvaluator():

    def __init__(self, model: CwaeModel):
        self.__model = model

        self.__evaluation_data = list()

        self.__z_dim = self.__model.z_dim
        self.__chunk_size = 128

    def build(self):

        latent_evaluators = list()
        latent_evaluators.append(MMDEvaluator(self.__z_dim))
        latent_evaluators.append(CWEvaluator(self.__z_dim))
        latent_evaluators.append(SWEvaluator(self.__z_dim))
        latent_evaluators.append(MardiaEvaluator())
        self.__latent_evaluators = latent_evaluators

        for latent_evaluator in self.__latent_evaluators:
            latent_evaluator.build()

        self.__tensor_rec_error_x = tf.placeholder(shape=[None] + self.__model.x_dim, dtype=tf.float32)
        self.__tensor_rec_error_y = tf.placeholder(shape=[None] + self.__model.x_dim, dtype=tf.float32)
        self.__tensor_rec_error = tf.reduce_sum(squared_euclidean_norm_reconstruction_error(self.__tensor_rec_error_x, 
                                                                                            self.__tensor_rec_error_y))

    def __encode_and_compute_rec_error(self, session, X):
        X_chunks = np.array_split(X, list(range(self.__chunk_size, len(X), self.__chunk_size)))
        z = np.ndarray([len(X), self.__z_dim])

        start_index = 0
        rec_err = 0

        for c in X_chunks:
            z1, y1 = self.__model.encode_decode(session, c)
            end_index = len(c) + start_index
            z[start_index:end_index, :] = z1
            feed_dict = {self.__tensor_rec_error_x: c, self.__tensor_rec_error_y: y1}
            rec_err += session.run(self.__tensor_rec_error, feed_dict=feed_dict)
            start_index = end_index

        return z, rec_err

    def encode(self, session, X):
        X_chunks = np.array_split(X, list(range(self.__chunk_size, len(X), self.__chunk_size)))
        z = np.ndarray([len(X), self.__z_dim])

        start_index = 0
        for c in X_chunks:
            z1, y1 = self.__model.encode_decode(session, c)
            end_index = len(c) + start_index
            z[start_index:end_index, :] = z1
            start_index = end_index

        return z

    def evaluate(self, session, X, epoch_number):
        print(f'Evaluation, test set count: {len(X)}')
        z, rec_err = self.__encode_and_compute_rec_error(session, X)
        rec_err = (1/len(X)) * rec_err

        result = {
            'rec_error': rec_err,
            'epoch_number': epoch_number
        }

        evaluation_results = list()
        for latent_evaluator in self.__latent_evaluators:
            evaluation_results += latent_evaluator.evaluate(session, z)

        for key, value in evaluation_results:
            result[key] = value

        self.__evaluation_data.append(result)

        return result, z

    def get_json(self) -> str:
        return json.dumps(self.__evaluation_data)
