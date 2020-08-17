import cv2
import numpy as np
import matplotlib.pyplot as plt
from process.cluster import Cluster, Direction


def find_hough_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 무의미한 선들을 없애기 위한 작업들
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = np.ones((4, 4), np.uint8)
    erode = cv2.erode(blur, kernel, iterations=1)

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

    clusters.sort(key=lambda object: len(object.item_list), reverse=True)

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
