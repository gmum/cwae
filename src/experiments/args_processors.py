import tensorflow as tf
import os

from architectures.celeba_architecture_provider import CelebaArchitectureProvider
from architectures.cifar10_architecture_provider import Cifar10ArchitectureProvider
from architectures.mnist_architecture_provider import MnistArchitectureProvider
from dataset_loaders import load_mnist_data, load_cifar10_data, create_cifar10_dataset, create_mnist_dataset


def get_optimizer_factory(optimizer_name, learning_rate: float):
    if optimizer_name == 'adam':
        return lambda: tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'gd':
        return lambda: tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError(f'Invalid optimizer configuration: {optimizer_name}, {learning_rate}')


def get_dataset_loader(dataset_name, batch_size: int):
    if dataset_name == 'celeba':
        raise ValueError(f'Write your own code to handle CELEBA dataset')
        # tr_images, ts_images = load_celeba_data('../data/CELEBA/datasets/')
        # return lambda: create_celeba_dataset(tr_images, ts_images, batch_size)
    elif dataset_name == 'cifar10':
        tr_images, ts_images = load_cifar10_data()
        return lambda: create_cifar10_dataset(tr_images, ts_images, batch_size)
    elif dataset_name == 'mnist':
        tr_images, ts_images = load_mnist_data()
        return lambda: create_mnist_dataset(tr_images, ts_images, batch_size)
    else:
        raise ValueError(f'Invalid dataset loader configuration: {dataset_name}, {batch_size}')


def get_architecture_provider(dataset_name: str):
    if dataset_name == 'mnist':
        return MnistArchitectureProvider()
    elif dataset_name == 'cifar10':
        return Cifar10ArchitectureProvider()
    elif dataset_name == 'celeba':
        return CelebaArchitectureProvider()
    else:
        raise ValueError(f'Invalid configuration {dataset_name}')


def prepare_base_dir(args):
    base_dir = f'../results/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    base_dir = f'{base_dir}{args.dataset_name}-{args.latent_size}/'

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    return base_dir
