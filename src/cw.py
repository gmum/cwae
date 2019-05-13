import tensorflow as tf
import math as m
from rec_errors import euclidean_norm_squared


def silverman_rule_of_thumb(N: int):
    return tf.pow(4/(3*N), 0.4)


def cw_1d(X, y=None):

    def N0(mean, variance):
        return 1.0/(tf.sqrt(2.0 * m.pi * variance)) * tf.exp((-(mean**2))/(2*variance))

    N = tf.cast(tf.shape(X)[0], tf.float32)
    if y is None:
        y = silverman_rule_of_thumb(N)

    A = tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1))
    return (1.0/(N*N)) * tf.reduce_sum(N0(A, 2*y)) + N0(0.0, 2.0 + 2*y) - (2/N) * tf.reduce_sum(N0(X, 1.0 + 2*y))


def cw_2d(X, y=None):
    def __phi(x):
        def __phi_f(s):
            t = s/7.5
            return tf.exp(-s/2) * (1 + 3.5156229*t**2 + 3.0899424*t**4 + 1.2067492*t**6 + 0.2659732*t**8
                                   + 0.0360768*t**10 + 0.0045813*t**12)

        def __phi_g(s):
            t = s/7.5
            return tf.sqrt(2/s) * (0.39894228 + 0.01328592*t**(-1) + 0.00225319*t**(-2) - 0.00157565*t**(-3)
                                   + 0.0091628*t**(-4) - 0.02057706*t**(-5) + 0.02635537*t**(-6) - 0.01647633*t**(-7)
                                   + 0.00392377*t**(-8))

        a = 7.5
        return __phi_f(tf.minimum(x, a)) - __phi_f(a) + __phi_g(tf.maximum(x, a))

    N = tf.cast(tf.shape(X)[0], tf.float32)
    if y is None:
        y = silverman_rule_of_thumb(N)

    A = 1/(N*N*tf.sqrt(y))
    B = 2.0/(N*tf.sqrt(y+0.5))

    A1 = euclidean_norm_squared(tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1)), axis=2)/(4*y)
    B1 = euclidean_norm_squared(X, axis=1)/(2+4*y)
    return 1/tf.sqrt(1+y) + A*tf.reduce_sum(__phi(A1)) - B*tf.reduce_sum(__phi(B1))


def cw(X, y=None):
    D = tf.cast(tf.shape(X)[1], tf.float32)
    N = tf.cast(tf.shape(X)[0], tf.float32)
    if y is None:
        y = silverman_rule_of_thumb(N)

    K = 1/(2*D-3)

    A1 = euclidean_norm_squared(tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1)), axis=2)
    A = (1/(N**2)) * tf.reduce_sum((1/tf.sqrt(y + K*A1)))

    B1 = euclidean_norm_squared(X, axis=1)
    B = (2/N)*tf.reduce_sum((1/tf.sqrt(y + 0.5 + K*B1)))

    return (1/tf.sqrt(1+y)) + A - B


def cw_choose(z_dim: int):
    if z_dim == 1:
        return cw_1d
    elif z_dim == 2:
        return cw_2d
    elif z_dim >= 20:
        return cw
    else:
        raise ValueError('Not defined for this latent dimension')


def cw_sampling(X, y=None):
    def phi_sampling(s, D):
        return tf.pow(1.0 + 4.0*s/(2.0*D-3), -0.5)

    D = tf.cast(tf.shape(X)[1], tf.float32)
    N = tf.cast(tf.shape(X)[0], tf.float32)
    D_int = tf.cast(D, tf.int32)
    N_int = tf.cast(N, tf.int32)
    if y is None:
        y = silverman_rule_of_thumb(N)

    YDistr = tf.contrib.distributions.MultivariateNormalDiag(loc=tf.zeros(D_int, tf.float32), 
                                                             scale_diag=tf.ones(D_int, tf.float32))
    Y = YDistr.sample(N_int)
    T = 1.0/(2.0*N*tf.sqrt(m.pi*y))

    A0 = euclidean_norm_squared(tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(X, 1)), axis=2)
    A = tf.reduce_sum(phi_sampling(A0/(4*y), D))

    B0 = euclidean_norm_squared(tf.subtract(tf.expand_dims(Y, 0), tf.expand_dims(Y, 1)), axis=2)
    B = tf.reduce_sum(phi_sampling(B0/(4*y), D))

    C0 = euclidean_norm_squared(tf.subtract(tf.expand_dims(X, 0), tf.expand_dims(Y, 1)), axis=2)
    C = tf.reduce_sum(phi_sampling(C0/(4*y), D))

    return T*(A + B - 2*C)
