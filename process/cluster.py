from enum import Enum
import numpy as np


class Direction(Enum):
    HORIZONTAL = 180
    VERTICAL = 90


class Cluster:
    def __init__(self, point1, point2, direction, image):
        self.direction = direction
        self.item_list = list()
        self.item_list.append((point1, point2))
        self.gamma = 64
        self.gradient_gamma = 0
        self.score = 0
        self.image = image

    def get_score(self):
        x1, x2 = self.get_mean_x()
        y1, y2 = self.get_mean_y()
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        center_point = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        line_p1 = np.array([x1, y1])
        line_p2 = np.array([x2, y2])
        distance = np.linalg.norm(np.cross(line_p2 - line_p1, line_p1 - center_point)) / np.linalg.norm(
            line_p2 - line_p1)

        return len(self.item_list) * 50 + length + \
               np.clip((np.maximum(self.image.shape[0], self.image.shape[1]) - distance), 0, 300)

    def append(self, point1, point2):
        self.item_list.append((point1, point2))

    def can_include(self, point1, point2):
        x1, x2 = self.get_mean_x()
        y1, y2 = self.get_mean_y()
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])

        # gradient = self.__get_gradient(point1, point2)
        # cur_gradient = self.__get_gradient(p1, p2)
        #
        # # 수평
        # elif abs(gradient - cur_gradient) < self.gradient_gamma:
        #     # 두 선의 거리를 측정하여

        if self.direction == Direction.VERTICAL:
            if abs(point1[0] - x1) < self.gamma and abs(point2[0] - x2) < self.gamma:
                return True

        else:
            if abs(point1[1] - y1) < self.gamma and abs(point2[1] - y2) < self.gamma:
                return True

        return False

    def __get_gradient(self, point1, point2):
        delta_x = abs(point1[0] - point2[0])
        delta_y = abs(point1[1] - point2[1])

        return delta_y / (delta_x + 1e-4)

    def get_mean_x(self):
        s1 = 0
        s2 = 0
        for item in self.item_list:
            s1 += item[0][0]
            s2 += item[1][0]
        return int(s1 / len(self.item_list)), int(s2 / len(self.item_list))

    def get_mean_y(self):
        s1 = 0
        s2 = 0
        for item in self.item_list:
            s1 += item[0][1]
            s2 += item[1][1]
        return int(s1 / len(self.item_list)), int(s2 / len(self.item_list))

