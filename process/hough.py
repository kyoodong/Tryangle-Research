import cv2
import numpy as np
import matplotlib.pyplot as plt
from process.cluster import Cluster, Direction
from skimage import feature


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
    skicanny = feature.canny(erode, sigma=3)
    skicanny = np.array(skicanny, dtype=np.uint8) * 255
    # canny = cv2.Canny(erode, 2500, 1500, apertureSize=5, L2gradient=True)

    lines = cv2.HoughLinesP(skicanny, 1, np.pi / 360, 100, minLineLength=200, maxLineGap=200)

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
                clusters.append(Cluster(point1, point2, Direction.HORIZONTAL, image))

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
                clusters.append(Cluster(point1, point2, Direction.VERTICAL, image))

    # 클러스터 평균 line이 중앙에 가까울 수록, 그 길이가 길 수록, 클러스터에 포함된 선의 수가 많을수록 우선순위가 높아지도록 정렬
    clusters.sort(key=lambda cluster: cluster.get_score(), reverse=True)

    count = 0
    for cluster in clusters:
        x1, x2 = cluster.get_mean_x()
        y1, y2 = cluster.get_mean_y()
        result.append([x1, y1, x2, y2])

        count += 1

        # 최대 3개의 유의미한 선을 추출
        if count == 3:
            break

    return result
