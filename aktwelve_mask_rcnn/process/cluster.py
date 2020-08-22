from enum import Enum


class Direction(Enum):
    HORIZONTAL = 180
    VERTICAL = 90


class Cluster:
    def __init__(self, point1, point2, direction):
        self.direction = direction
        self.item_list = list()
        self.item_list.append((point1, point2))
        self.gamma = 20

    def append(self, point1, point2):
        self.item_list.append((point1, point2))

    def can_include(self, point1, point2):
        if self.direction == Direction.VERTICAL:
            x1, x2 = self.get_mean_x()
            if abs(point1[0] - x1) < self.gamma and abs(point2[0] - x2) < self.gamma:
                return True
            return False
        else:
            y1, y2 = self.get_mean_y()
            if abs(point1[1] - y1) < self.gamma and abs(point2[1] - y2) < self.gamma:
                return True
            return False

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

