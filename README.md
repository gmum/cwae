# Repository info
This repository contains an implementation of Cramer-Wold AutoEncoder(CWAE), proposed by [Jacek Tabor, Szymon Knop, Przemysław Spurek, Igor Podolak, Marcin Mazur, Stanisław Jastrzębski (2018)](https://arxiv.org/abs/1805.09235).

# Contents of the repository
```
|-- Mnist 2D.ipynb - Jupyter Notebook with trivial implementation presenting basic features
|-- src/ - contains an implementation of CWAE allowing to reproduce experiments from the original paper
|---- train_models.py - the starting point for experiments
|---- cw.py - implementation of various versions CW Distance
|---- architectures/ - files containing architectures used proposed in the paper
|---- evaluation/ - implementation of metrics used to evaluate and compare models
|-- results/ - directory that will be created in order to store the results of conducted experiments
```
# Conducting the experiments
In order to reproduce CWAE results on MNIST and CIFAR-10 experiments as described in the original paper execute the following commands:

    python train_models.py cwae mnist 500 20
    python train_models.py cwae cifar10 500 64

## Other options
The code allows manipulating some of the parameters(for example using other versions of the model, changing learning rate values or optimizer types) for more info see the list of available arguments in src/train_models.py file

# Datasets
The repository allows reproducing experiments on MNIST and CIFAR10 dataset. In order to run experiments on CelebA dataset one must download it and import in a similar manner as other datasets by implementing load_celeba_data in src/dataset_loaders.py

# Environment
We created the repository using the following configuration:
- python 3.7.2
- tensorflow 1.13.1
- numpy 1.16.2
- matplotlib 3.0.3

# Additional links
To compute FID Score we used the code from:
- https://github.com/bioinf-jku/TTUR

To evaluate WAE model we used the code from:
- https://github.com/tolstikhin/wae

CelebA dataset is available here:
- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
