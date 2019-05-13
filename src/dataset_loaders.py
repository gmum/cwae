import tensorflow as tf
import numpy as np

from experiments.canvas_fillers import RgbCanvasFiller, GrayscaleCanvasFiller


class Dataset:

    def __init__(self, train, train_images: list, validation_images: list, x_dim, batch_size: int, canvas_filler):
        self.train = train
        self.train_images = train_images
        self.train_images_count = len(train_images)
        self.validation_images = validation_images
        self.x_dim = x_dim
        self.__iterator = self.train.make_initializable_iterator()
        self.batch_size = batch_size
        self.canvas_filler = canvas_filler

    def get_iterator(self) -> tf.data.Iterator:
        return self.__iterator

    def initialize_iterator(self, tf_session):
        tf_session.run(self.__iterator.initializer)


def load_cifar10_data():
    print('Loading CIFAR10 dataset')
    (tr_images, _), (ts_images, _) = tf.keras.datasets.cifar10.load_data()
    print('Loading completed')
    return tr_images.astype(np.float32) / 255.0, ts_images.astype(np.float32) / 255.0


def load_mnist_data():
    print('Loading MNIST dataset')
    (tr_images, _), (ts_images, _) = tf.keras.datasets.mnist.load_data()
    print('Loading completed')
    return tr_images.astype(np.float32) / 255.0,  ts_images.astype(np.float32) / 255.0


def load_celeba_data():
    print('Loading CELEBA dataset')
    # Implement loading method: (tr_images, _), (ts_images, _) = ...
    # print('Loading completed')
    # return tr_images.astype(np.float32) / 255.0,  ts_images.astype(np.float32) / 255.0


def create_mnist_dataset(tr_images, ts_images, batch_size):
    return create_dataset(tr_images, ts_images, [28, 28], batch_size, GrayscaleCanvasFiller())


def create_cifar10_dataset(tr_images, ts_images, batch_size):
    return create_dataset(tr_images, ts_images, [32, 32, 3], batch_size, RgbCanvasFiller())


def create_dataset(tr_images, ts_images, x_dim, batch_size, canvas_filler):
    train_dataset = tf.data.Dataset.from_tensor_slices(tr_images)
    train_dataset = train_dataset.shuffle(buffer_size=30000)
    train_dataset = train_dataset.batch(batch_size).prefetch(1)

    return Dataset(train_dataset, tr_images, ts_images[0:], x_dim, batch_size, canvas_filler)
