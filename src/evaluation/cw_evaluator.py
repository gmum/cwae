import tensorflow as tf
import numpy as np
from cw import cw_choose


class CWEvaluator:

    def __init__(self, z_dim: int, samples_limit: int = 2000, repeats_count: int = 5):
        self.__z_dim = z_dim
        self.__samples_limit = samples_limit
        self.__repeats_count = repeats_count

    def build(self):
        self.__tensor_z_encoded = tf.placeholder(shape=np.append([None], self.__z_dim), dtype=tf.float32)
        self.__tensor_cw_index = cw_choose(self.__z_dim)(self.__tensor_z_encoded)

    def __compute_cw_distance(self, session: tf.Session, z):
        feed_dict = {
            self.__tensor_z_encoded: z,
        }
        return session.run(self.__tensor_cw_index, feed_dict=feed_dict)

    def evaluate(self, session: tf.Session, z):
        print('Computing CW')

        if z.shape[0] > self.__samples_limit:
            cw_value = 0.0
            for _ in range(self.__repeats_count):
                index = np.random.choice(z.shape[0], self.__samples_limit, replace=False)
                latent = z[index]
                cw_value += self.__compute_cw_distance(session, latent)
            cw_value = cw_value / self.__repeats_count
        else:
            cw_value = self.__compute_cw_distance(session, z)

        return [('cw', cw_value)]
