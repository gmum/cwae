from cwae_model import CwaeModel
from cw import cw_choose, cw_sampling


class CWAEFactory:

    def __init__(self, rec_error, gamma=None) -> CwaeModel:
        self.__rec_error = rec_error
        self.__gamma = gamma

    def create(self, dataset, z_dim):
        x_dim = dataset.x_dim
        iterator = dataset.get_iterator()

        chosen_cw_method = cw_choose(z_dim)
        normality_index = lambda tensor_z: chosen_cw_method(tensor_z, self.__gamma)

        print(f'Creating CWAE {x_dim}, {z_dim} with gamma={self.__gamma}')
        return CwaeModel(x_dim, z_dim, iterator, self.__rec_error, normality_index)


class CWAESamplingFactory:

    def __init__(self, rec_error, gamma=None) -> CwaeModel:
        self.__rec_error = rec_error
        self.__gamma = gamma

    def create(self, dataset, z_dim):
        x_dim = dataset.x_dim
        iterator = dataset.get_iterator()

        if z_dim < 20:
            raise ValueError('Not defined for this latent dimension')
        normality_index = lambda tensor_z: cw_sampling(tensor_z, self.__gamma)

        print(f'Creating CWAE Sampling {x_dim}, {z_dim} with gamma={self.__gamma}')
        return CwaeModel(x_dim, z_dim, iterator, self.__rec_error, normality_index)
