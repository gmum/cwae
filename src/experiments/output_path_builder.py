import os


class OutputPathBuilder:
    def __init__(self, base_dir):
        self.__base_dir = base_dir
        self.__prefix = ''

    def set_prefix(self, prefix):
        self.__prefix = prefix

    def get_base_dir(self):
        return self.__base_dir

    def get_path(self, name: str):
        if not os.path.exists(self.__base_dir):
            os.mkdir(self.__base_dir)
        return os.path.join(self.__base_dir, f'{self.__prefix}_{name}')
