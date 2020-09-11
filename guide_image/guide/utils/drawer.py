from matplotlib import pyplot as plt
import numpy as np
import cv2

def draw_line(src, lines, color=(0, 0, 255), thickness=1):
    '''
    
    :param src: 선을 그릴 이미지 배열
    :param lines: 선분의 양 끝의 점의 정보를 가진 배열, (N, 4)
    :param color: 선분의 색
    :param thickness: 선분의 두께
    :return: 선이 그려진 이미지
    '''
    dst = src.copy()

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(dst, (x1, y1), (x2, y2), color, thickness=thickness)

    return dst


def draw_line_cluster(src, lines, clusters):
    dst = src.copy()

    colors = get_random_color(max(clusters))

    if lines is not None:
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i]
            cluster_idx = clusters[i]

            cv2.line(dst, (x1, y1), (x2, y2), color=colors[cluster_idx - 1], thickness=1)

    return dst

def display(title, image):
    plt.title(title)

    if len(image.shape) <= 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.xticks([]), plt.yticks([])

    plt.show()

def get_random_color(color_len):

    colors = []
    for i in range(color_len):
        colors.append(list(np.random.random(size=3) * 255))

    return colors