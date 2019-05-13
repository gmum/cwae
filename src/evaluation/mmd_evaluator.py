import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import MultivariateNormalDiag


def mmd_penalty(sample_qz, sample_pz, kernel: str = 'IMQ'):
    n = tf.cast(tf.shape(sample_qz)[0], tf.float32)
    d = tf.cast(tf.shape(sample_qz)[1], tf.float32)

    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)

    if kernel == 'IMQ':
        sigma2_p = 1. ** 2
        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * tf.matmul(sample_pz, sample_pz, transpose_b=True)

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * tf.matmul(sample_qz, sample_qz, transpose_b=True)

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        Cbase = 2. * d * sigma2_p

        stat = 0.
        TempSubtract = 1. - tf.eye(n)
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz) + C / (C + distances_pz)
            res1 = tf.multiply(res1, TempSubtract)
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat
    else:
        ValueError(f'Unknown kernel function: {kernel}')


class MMDEvaluator:

    def __init__(self, z_dim, repeats_count: int = 3, samples_limit: int = 2000):
        self.__z_dim = z_dim
        self.__repeats_count = repeats_count
        self.__samples_limit = samples_limit

    def build(self):
        self.__tensor_z_encoded = tf.placeholder(shape=np.append([None], self.__z_dim), dtype=tf.float32)
        self.__distr = MultivariateNormalDiag(loc=tf.zeros(self.__z_dim), scale_diag=tf.ones(self.__z_dim))
        self.__tensor_z_sampled = self.__distr.sample(tf.shape(self.__tensor_z_encoded)[0])
        self.__tensor_mmd_penalty = mmd_penalty(self.__tensor_z_encoded, self.__tensor_z_sampled)

    def __compute_wae_distance(self, session, latent):
        mmd_penalty_sum = 0
        feed_dict = {
            self.__tensor_z_encoded: latent,
        }

        for _ in range(self.__repeats_count):
            mmd_penalty_sum += session.run(self.__tensor_mmd_penalty, feed_dict)

        avg_mmd_penalty = mmd_penalty_sum / self.__repeats_count
        return avg_mmd_penalty

    def evaluate(self, session, z):
        print('Computing MMD')
        if z.shape[0] > self.__samples_limit:
            index = np.random.choice(z.shape[0], self.__samples_limit, replace=False)
            wae_distance = self.__compute_wae_distance(session, z[index])
        else:
            wae_distance = self.__compute_wae_distance(session, z)

        return [('wae_distance', wae_distance)]
