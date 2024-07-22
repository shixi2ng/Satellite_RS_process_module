import copy
import random
import numpy as np
import os

import pandas as pd


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
        self._maximum_node_perlvl = 3
        self._minimum_river_width = 4

        # Define work env
        if work_env is None or not os.path.exists(work_env):
            self.work_env = os.path.join(os.getcwd() + 'rivercentreline_sample\\')
        else:
            self.work_env = os.path.join(work_env + 'rivercentreline_sample\\')

        # Define the dimension
        if not isinstance(dimension, tuple):
            raise TypeError('The dimension should be under the tuple type!')
        elif dimension[0] != dimension[1]:
            raise ValueError('Only square matrix is valid under current version!')
        elif dimension[0] not in self._support_dimension:
            raise ValueError(f'The dimension {str(dimension)} is not supported!')
        else:
            self.dimension = dimension[0]

        # Define the network_level
        self._max_network_lvl = np.floor(self.dimension/(self._minimum_river_width * 2 * 16))
        if network_lvl is None:
            self.network_lvl = 4
        elif not isinstance(network_lvl, int):
            raise TypeError('The network level should be under the tuple type!')
        else:
            self.network_lvl = max(min(self._max_network_lvl, network_lvl), 1)

        # Define key variable
        self.seed_amount = int(((self.dimension / 2) ** 2) * 4)
        self.network_width = [2 * (self._node_leaf ** (self.network_lvl - _)) + 1 for _ in range(self.network_lvl + 1)]

    def _generate_sample_base(self, seed_amount=None):

        # Create the seed as the sequence shown below
        # 1 Determine the network level range shown in the sample, for example (0 to 1), (1 to 2), (1 to 3) etc.
        # 2 Determine the number of nodes within each level
        # 3

        if seed_amount is None or seed_amount < self.seed_amount:
            print('Default seed amount will be applied!')
        else:
            self.seed_amount = seed_amount

        # Create network level combination
        for seed_ in range(self.seed_amount):

            # Generate the node
            node_beg_lvl = random.choice([_ for _ in range(self.network_lvl - 1)])
            node_end_lvl = random.choice([_ for _ in range(node_beg_lvl, self.network_lvl - 1)])

            # print(f'{str(node_beg_lvl)}   {str(node_end_lvl)}')
            node_list, node_lvl_list = [], [_ for _ in range(node_beg_lvl, node_end_lvl + 1)]
            for node_lvl in node_lvl_list:
                node_num = random.choice(range(1, self._maximum_node_perlvl + 1))
                node_list.extend([node_lvl * 1 + _ * 0.1 for _ in range(node_num)])

            # Generate the start and end list
            start_list, end_list = [], []
            for node_ in range(node_beg_lvl, node_end_lvl + 1):
                if node_ == node_beg_lvl:
                    start_branch_num = int(np.sum(np.floor(np.array(node_list)) == node_))
                else:
                    start_branch_num = int(np.ceil(max(np.sum(np.floor(np.array(node_list)) == node_) - np.sum(np.floor(np.array(node_list)) == node_ - 1) * self._node_leaf, 0)))
                start_list.extend([10 + node_ + _ * 0.1 for _ in range(start_branch_num)])

                if node_ != node_end_lvl:
                    end_branch_num = int(np.sum(np.floor(np.array(node_list)) == node_) * self._node_leaf - np.sum(np.floor(np.array(node_list)) == node_ + 1))
                else:
                    end_branch_num = int(np.sum(np.floor(np.array(node_list)) == node_) * self._node_leaf)
                end_list.extend([20 + node_ + 1 + _ * 0.1 for _ in range(end_branch_num)])
            print(f'node:{str(node_list)}; start_list:{str(start_list)}; end_list:{str(end_list)}')

            # Generate all the lines and sequence them randomly (based on the node relationship)
            input_line_list = []
            for node_ in range(node_beg_lvl, node_end_lvl + 2):
                input_line_list.extend([[st_node_] for st_node_ in start_list if np.floor(st_node_) == 10 + node_])
                curr_lvl_node = [_ for _ in node_list if np.floor(_) == node_]
                curr_lvl_node.extend([_ for _ in end_list if np.floor(_) == 20 + node_])
                random.shuffle(curr_lvl_node)

                if len([_ for _ in input_line_list if _[-1] < 20]) != len(curr_lvl_node):
                    raise Exception('Code error!')

                future_line_list = []
                count = 0
                while count < len(input_line_list):
                    input_line = input_line_list[count]
                    if input_line[-1] < 20:
                        curr_lvl = curr_lvl_node[0]
                        curr_lvl_node.remove(curr_lvl)
                        input_line.append(curr_lvl)

                        if curr_lvl in end_list:
                            future_line_list.append(copy.deepcopy(input_line))
                        else:
                            future_line_list.append(copy.deepcopy(input_line))
                            future_line_list.append(copy.deepcopy(input_line))
                    else:
                        future_line_list.append(copy.deepcopy(input_line))
                    count += 1
                input_line_list = copy.deepcopy(future_line_list)
            print(str(input_line_list))

            # Assemble all the line into the space (Pseudo-random)
            line_coord_df = pd.DataFrame({'st_node': [], 'end_node': [], 'line_coord': []})
            node_coord_df = pd.DataFrame({'node_id': [], 'coord': [], 'width': []})
            boundary_len = int(self.dimension / 2)
            boundary_offset = [[1, 0], [0, 1], [-1, 0], [0, -1]]
            boundary_origin = [[0, 0], [boundary_len, 0], [boundary_len, boundary_len], [0, boundary_len]]
            boundary_list = [0, self.dimension * 2]
            st_count, ed_count = 0, 0
            for st_node in start_list:

                st_range = [boundary_list[0], int(boundary_list[0] + (boundary_list[1] - boundary_list[0]) / (2 * (len(start_list) - st_count)))]
                st_count += 1
                st_node_pos = random.choice(range(st_range[0], st_range[1]))
                st_node_bound = int(np.floor(2 * st_node_pos / self.dimension))
                st_node_offset = st_node_pos - st_node_bound * self.dimension / 2
                st_coord = [boundary_origin[st_node_bound][0] + st_node_offset * (boundary_offset[st_node_bound][0]), boundary_origin[st_node_bound][1] + st_node_offset * (boundary_offset[st_node_bound][1])]
                st_width = int(self.network_width[int(np.floor(st_node) - 10)] * random.choice(range(8000, 12000)) / 10000)
                line_under_st = [_ for _ in input_line_list if _[0] == st_node]
                boundary_list = [boundary_list[0] + st_node_pos + int(np.floor(st_width / 2)), min(boundary_list[1], self.dimension * 2 - (int(np.floor(st_width / 2)) - st_node_pos))]
                node_coord_df = pd.concat([node_coord_df, pd.DataFrame({'node_id': [st_node], 'coord': [st_coord], 'width': [st_width]})], ignore_index = True)

                for line in line_under_st:
                    ed_range = [int(boundary_list[1] - (boundary_list[1] - boundary_list[0]) * (ed_count + 1) / (2 * len(end_list))), boundary_list[1]]
                    ed_count += 1
                    ed_node_pos = random.choice(range(ed_range[0], ed_range[1]))
                    ed_node_bound = int(np.floor(2 * ed_node_pos / self.dimension))
                    ed_node_offset = ed_node_pos - ed_node_bound * self.dimension / 2
                    ed_coord = [boundary_origin[ed_node_bound][0] + ed_node_offset * (boundary_offset[ed_node_bound][0]),
                                boundary_origin[ed_node_bound][1] + ed_node_offset * (boundary_offset[ed_node_bound][1])]




if __name__ == '__main__':
    rcentre = RiverCentreline_sample((256, 256))
    rcentre._generate_sample_base()
