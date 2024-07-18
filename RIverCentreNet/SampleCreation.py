import numpy as np
import os

class RiverCentreline_sample(object):

    def __init__(self, dimension: tuple, network_lvl:int=None, work_env: str=None):
        """

        :param dimension: Dimension denotes the size of input images, the recommendation is 256 * 256 or 512 * 512
        :param network_level: Network level represents the complexity of river networks, which also
        :param work_env: The folder used to generate the river network sample data
        """

        # Define init var
        self._support_dimension = [128, 256, 512, 1024]
        self._node_leaf = 2
        self._minimum_river_width = 4

        # Define work env
        if self.work_env is None or not os.path.exists(work_env):
            self.work_env = os.path.join(os.getcwd() + 'rivercentreline_sample\\')
        else:
            self.work_env = os.path.join(work_env + 'rivercentreline_sample\\')

        # Define the dimension
        if not isinstance(dimension, tuple):
            raise TypeError('The dimension should be under the tuple type!')
        elif dimension[0] != dimension[1]:
            raise ValueError('Only square matrix is valid under current version!')
        elif dimension not in self._support_dimension:
            raise ValueError(f'The dimension {str(dimension)} is not supported!')
        else:
            self.dimension = dimension[0]

        # Define the network_level
        self._max_network_lvl = np.floor(self.dimension/(self._minimum_river_width * 2 * 16)) - 1
        if not isinstance(network_lvl, int):
            raise TypeError('The network level should be under the tuple type!')
        else:
            self.network_lvl = max(min(self._max_network_lvl, network_lvl), 1)

        # Define key variable
        self.seed_amount = int(((self.dimension / 2) ** 2) * 4)

    def _generate_sample_base(self, seed_amount = None):

        # Create the seed as the sequence shown below
        # 1 Determine the network level range shown in the sample, for example (0 to 1), (1 to 2), (1 to 3) etc.
        # 2 Determine the number of nodes within each level
        # 3

        if seed_amount is None or seed_amount < self.seed_amount:
            print('Default seed amount will be applied!')
        else:
            self.seed_amount = seed_amount

        # Create network level combination
        for _ in range(seed_amount):


