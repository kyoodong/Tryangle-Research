import cv2
import numpy as np
from guide.process import ciede2000 as color_diff


def find_similar_by_color(query_color, other_colors, count=3):
    if count > len(other_colors):
        count = len(other_colors)

    diff = diff_color(query_color, other_colors)
    diff_arg = np.argsort(diff)

    return diff_arg[:count]

def diff_color(query_color, other_colors):
    diff = []
    ciede2000 = color_diff.ciede2000
    rgb2lab = color_diff.rgb2lab

    print("Query", query_color)
    print("Other", other_colors)

    for pallet in other_colors:
        color_diff_sum = 0
        for o_color in pallet:
            color_diff_sum += np.min([ciede2000(rgb2lab(q_color), rgb2lab(o_color)) for q_color in query_color])
        diff.append(color_diff_sum)
    return np.array(diff)

#########################################
# 주요 색 찾기
#########################################
def get_dominant_color(img, n_color=4, image_processing_size=(32,32)):
    image = cv2.resize(img, image_processing_size)
    pixels = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    a, labels, pallets = cv2.kmeans(pixels, n_color, None, criteria, 10, flags)
    b, counts = np.unique(labels, return_counts=True)
    counts_argsort = np.argsort(counts)[::-1]

    return pallets[counts_argsort]
