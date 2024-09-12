import copy
import random
import numpy as np
import os
from shapely.geometry import Point, Polygon
import pandas as pd


def create_buffer(point, radius, grid_size):
    # 创建一个空的布尔数组，尺寸为grid_size
    buffer_array = np.zeros(grid_size, dtype=np.uint8)

    # 获取网格的所有点的索引
    y, x = np.ogrid[:grid_size[0], :grid_size[1]]

    # 计算每个点到中心点的距离
    distance = np.sqrt((x - point[1]) ** 2 + (y - point[0]) ** 2)

    # 将距离小于或等于半径的位置设置为True
    buffer_array[distance <= radius] = True

    return buffer_array


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

    def _1dto2d_boundary_coord(self, pos, dimension, clockwise=False):
        boundary_offset = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        boundary_origin = [[0, 0], [dimension, 0], [dimension, dimension], [0, dimension]]
        node_bound = int(np.floor(pos / dimension))
        node_offset = pos - node_bound * dimension
        return [boundary_origin[node_bound][0] + node_offset * (boundary_offset[node_bound][0]), boundary_origin[node_bound][1] + node_offset * (boundary_offset[node_bound][1])]

    def _random_point_in_polygon(poly: Polygon):
        minx, miny, maxx, maxy = poly.bounds
        while True:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.within(p):
                return p

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

            boundary_list = [0, self.dimension * 2]

            # Define the boundary shp list
            boundary_outside_list = [int(self.dimension/2), int(self.dimension/2), int(self.dimension * 3/2)]
            boundary_inside_list = [0]

            # Define the river array
            river_arr = np.zeros([int(self.dimension/2), int(self.dimension/2)], dtype=np.uint8)

            st_count, ed_count = 0, 0
            for st_node in start_list:
                st_range = [boundary_list[0], int(boundary_list[0] + (boundary_list[1] - boundary_list[0]) / (2 * (len(start_list) - st_count)))]
                st_count += 1
                st_node_pos = random.choice(range(st_range[0], st_range[1]))
                if st_node_pos == 0:
                    st_node_pos = 1
                elif np.mod(st_node_pos, self.dimension/2) == 0:
                    st_node_pos = st_node_pos - 1
                st_coord = self._1dto2d_boundary_coord(st_node_pos, int(self.dimension / 2))
                st_width = int(self.network_width[int(np.floor(st_node) - 10)] * random.choice(range(8000, 12000)) / 10000)
                line_under_st = [_ for _ in input_line_list if _[0] == st_node]
                boundary_list = [boundary_list[0] + st_node_pos + int(np.floor(st_width / 2)), min(boundary_list[1], self.dimension * 2 - (int(np.floor(st_width / 2)) - st_node_pos))]

                for line in line_under_st:
                    ed_node = line[-1]
                    ed_range = [int(boundary_list[1] - (boundary_list[1] - boundary_list[0]) * (ed_count + 1) / (2 * len(end_list))), boundary_list[1]]
                    ed_count += 1
                    ed_node_pos = random.choice(range(ed_range[0], ed_range[1]))
                    if ed_node_pos == 0:
                        ed_node_pos = 1
                    elif np.mod(ed_node_pos, self.dimension / 2) == 0:
                        ed_node_pos = ed_node_pos - 1
                    ed_coord = self._1dto2d_boundary_coord(ed_node_pos, int(self.dimension / 2))
                    ed_width = int(self.network_width[int(np.floor(ed_node) - 20)] * random.choice(range(8000, 12000)) / 10000)

                    node_inside_polygon = [boundary_inside_list[0]]
                    for _ in boundary_outside_list:
                        if _ < st_node_pos:
                            node_inside_polygon.append(_)

                    node_inside_polygon.extend([st_node_pos, ed_node_pos])
                    for _ in boundary_outside_list:
                        if _ > ed_node_pos:
                            node_inside_polygon.append(_)

                    boundary_outside_list = [_ for _ in boundary_outside_list if _ not in node_inside_polygon]
                    boundary_inside_list = [st_node_pos, ed_node_pos]
                    path = []
                    previous_point_record = False
                    for node_ in line:
                        if node_ in list(node_coord_df['node_id']):
                            previous_point_record = True
                        elif 10 <= node_ < 20:
                            node_coord_df = pd.concat([node_coord_df, pd.DataFrame({'node_id': [st_node], 'coord': [st_coord], 'width': [st_width]})], ignore_index=True)
                            node_coord_inside_polygon = [self._1dto2d_boundary_coord(_, int(self.dimension / 2)) for _ in node_inside_polygon]
                            replace_node = st_coord
                        elif node_ >= 20:
                            intersect_with_other_line, count = True, 0
                            previous_point = line[line.index(node_) - 1]
                            while intersect_with_other_line and count < 1000:
                                point_coord = Polygon(node_coord_inside_polygon)
                                if previous_point_record:
                                    line_width_ = int(self.network_width[int(node_)] * random.choice(range(8000, 12000)) / 10000)
                                else:

                                    other_line = [_ for _ in line if previous_point in _ and _ != line][0]
                                    other_point = other_line[other_line.index(previous_point) + 1]
                                    line_width_ = node_coord_df[node_coord_df['node_id'] == previous_point]['width'] - node_coord_df[node_coord_df['node_id'] == other_point]['width']
                                buffer_temp = create_buffer(point_coord, line_width_/2, [int(self.dimension/2), int(self.dimension/2)])
                                intersect_with_other_line = np.sum(np.logical_and(buffer_temp, river_arr)) > 0
                                count += 1

                            new_path, new_river_arr = self._random_path_(previous_point, point_coord, line_width_, river_arr)
                            if path[-1] == new_path[0]:
                                path.pop()
                                path.extend(new_path)
                            river_arr = np.logical_or(new_river_arr, river_arr)
                            node_coord_df = pd.concat([node_coord_df, pd.DataFrame({'node_id': [ed_node], 'coord': [ed_coord], 'width': [ed_width]})], ignore_index=True)
                        else:
                            intersect_with_other_line, count = True, 0
                            previous_point = line[line.index(node_) - 1]
                            while intersect_with_other_line and count < 1000:
                                point_coord = Polygon(node_coord_inside_polygon)
                                if previous_point_record:
                                    line_width_ = int(self.network_width[int(node_)] * random.choice(range(8000, 12000)) / 10000)
                                else:

                                    other_line = [_ for _ in line if previous_point in _ and _ != line][0]
                                    other_point = other_line[other_line.index(previous_point) + 1]
                                    line_width_ = node_coord_df[node_coord_df['node_id'] == previous_point]['width'] - node_coord_df[node_coord_df['node_id'] == other_point]['width']
                                buffer_temp = create_buffer(point_coord, line_width_/2, [int(self.dimension/2), int(self.dimension/2)])
                                intersect_with_other_line = np.sum(np.logical_and(buffer_temp, river_arr)) > 0
                                count += 1

                            new_path, new_river_arr = self._random_path_(previous_point, point_coord, line_width_, river_arr)
                            if path[-1] == new_path[0]:
                                path.pop()
                                path.extend(new_path)
                            river_arr = np.logical_or(new_river_arr, river_arr)

                            node_coord_inside_polygon = [_ if _ != replace_node else point_coord for _ in node_coord_inside_polygon]
                            node_coord_df = pd.concat([node_coord_df, pd.DataFrame({'node_id': [node_], 'coord': [point_coord], 'width': [line_width_]})], ignore_index=True)

    def _random_path_(self, start, end, width, river):
        new_river = np.zeros_like(river, dtype=np.uint8)
        new_path = []
        stack = [start]
        directions = []
        dx = end[1] - start[1]
        dy = end[0] - start[0]

        # 根据dx和dy的符号决定主要和次要的移动方向
        horizontal = (0, 1) if dx > 0 else (0, -1)
        vertical = (1, 0) if dy > 0 else (-1, 0)

        # 添加主要方向和次要方向
        directions.append(horizontal)
        if dx != 0:
            directions.append(vertical)
            directions.append((vertical[0], horizontal[1]))

        while stack:
            current = stack[-1]
            new_path.append(current)
            buffer_temp = create_buffer(current, width/2, [river.shape[0], river.shape[1]])
            new_river = np.logical_or(buffer_temp, new_river).astype(np.uint8)

            if current == end:
                return new_path, new_river

            possible_moves = []
            random.shuffle(directions)  # 随机化方向以增加路径的随机性
            for direction in directions:
                next_point = (current[0] + direction[0], current[1] + direction[1])
                buffer_next_point = create_buffer(next_point, width/2, [river.shape[0], river.shape[1]])
                if (min(start[0], end[0]) <= next_point[0] <= max(start[0], end[0]) and min(start[1], end[1]) <= next_point[1] <= max(start[1], end[1]) and
                        next_point not in new_path and np.sum(np.logical_and(buffer_next_point, river)) == 0):
                    possible_moves.append(next_point)

            if possible_moves:
                stack.append(random.choice(possible_moves))
            else:
                stack.pop()  # 无法前进时回溯

        return new_path, new_river


if __name__ == '__main__':
    rcentre = RiverCentreline_sample((256, 256))
    rcentre._generate_sample_base()
