import cv2
import numpy as np
import matplotlib.pyplot as plt
from process.cluster import Cluster, Direction


def find_hough_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 무의미한 선들을 없애기 위한 작업들
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.filter2D(gray, -1, kernel)
    kernel = np.ones((4, 4), np.uint8)
    erode = cv2.erode(gray, kernel, iterations=1)

    # 선 추출
    canny = cv2.Canny(erode, 2500, 1500, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 360, 100, minLineLength=200, maxLineGap=200)

    if lines is None:
        return None

    # 수평선에서는 좌우, 수직선에서는 상하의 점의 수평 차이가 gamma 미만이면 수평, 수직선이라 판단하고 가이드를 제공
    gamma = 25
    clusters = list()
    result = list()

    for line in lines:
        # HoughP
        x1, y1, x2, y2 = line[0]

        # 수평선만 검출
        if abs(y1 - y2) < gamma:
            point1 = (x1, y1)
            point2 = (x2, y2)
            is_process = False
            for cluster in clusters:
                if cluster.can_include(point1, point2):
                    cluster.append(point1, point2)
                    is_process = True
                    break

            if not is_process:
                clusters.append(Cluster(point1, point2, Direction.HORIZONTAL))

        # 수직선만 검출
        elif abs(x1 - x2) < gamma:
            point1 = (x1, y1)
            point2 = (x2, y2)
            is_process = False
            for cluster in clusters:
                if cluster.can_include(point1, point2):
                    cluster.append(point1, point2)
                    is_process = True
                    break

            if not is_process:
                clusters.append(Cluster(point1, point2, Direction.VERTICAL))

    def sort_rule(cluster):
        x1, x2 = cluster.get_mean_x()
        y1, y2 = cluster.get_mean_y()
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        center_point = (image.shape[1] // 2, image.shape[0] // 2)
        line_p1 = np.array([x1, y1])
        line_p2 = np.array([x2, y2])
        distance = np.linalg.norm(np.cross(line_p2 - line_p1, line_p1 - center_point)) / np.linalg.norm(
            line_p2 - line_p1)

        return len(cluster.item_list) * 50 + length +\
            np.clip((np.maximum(image.shape[0], image.shape[1]) - distance), 0, 300)

    # 길이나 위치에 따른 선의 우선순위 보정이 필요함
    # 단순 개수만으로는 유의미한 선 검출이 어려움
    clusters.sort(key=sort_rule, reverse=True)

    count = 0
    for cluster in clusters:
        print(len(cluster.item_list))
        x1, x2 = cluster.get_mean_x()
        y1, y2 = cluster.get_mean_y()
        result.append([x1, y1, x2, y2])

        count += 1
        if count == 3:
            break

    return result
