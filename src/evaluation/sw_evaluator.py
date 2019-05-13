import tensorflow as tf
import numpy as np


def swae_normality_index(projected_latent, theta, z_dim):
    n = tf.cast(tf.shape(projected_latent)[0], tf.int32)

    dist = tf.distributions.Normal(np.zeros(z_dim, dtype=np.float32), np.ones(z_dim, dtype=np.float32))
    sample = dist.sample(n)

    # Let projz be the projection of the $q_Z$ samples
    projz = tf.keras.backend.dot(sample, tf.transpose(theta))
    # Calculate the Sliced Wasserstein distance by sorting
    # the projections and calculating the L2 distance between

    transposed_projected_latent = tf.transpose(projected_latent) 
    transpose_projected_sample = tf.transpose(projz)

    W2 = (tf.nn.top_k(transposed_projected_latent, k=n).values -
          tf.nn.top_k(transpose_projected_sample, k=n).values)**2

    return W2


def SWAE_cost(tensor_z, z_dim):
    randomed_normal = tf.random_normal(shape=(50, z_dim))
    theta = randomed_normal / tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(randomed_normal), axis=1)), (-1, 1))

    # Let projae be the projection of the encoded samples
    projae = tf.keras.backend.dot(tensor_z, tf.transpose(theta))

    normality_test_result = swae_normality_index(projae, theta, z_dim)
    return tf.reduce_mean(normality_test_result)


class SWEvaluator:

    def __init__(self, z_dim, repeats=10):
        self.__z_dim = z_dim
        self.__repeats = repeats

    def build(self):
        self.__tensor_z_encoded = tf.placeholder(shape=np.append([None], self.__z_dim), dtype=tf.float32)
        self.__tensor_sw_distance = SWAE_cost(self.__tensor_z_encoded, self.__z_dim)

    def evaluate(self, session: tf.Session, z):
        print('Computing SW Distance')
        sw_distance = 0
        for _ in range(self.__repeats):
            feed_dict = {
                self.__tensor_z_encoded: z,
            }
            sw_distance += session.run(self.__tensor_sw_distance, feed_dict)

        avg_sw_distance = sw_distance / self.__repeats
        return [('sw_distance', avg_sw_distance)]