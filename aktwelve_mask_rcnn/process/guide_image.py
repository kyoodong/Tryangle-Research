import cv2
import numpy as np


def find_dominant_colors(image):
    pixels = np.float32(image.reshape(-1, 3))
    n_color = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    a, labels, pallets = cv2.kmeans(pixels, n_color, None, criteria, 10, flags)
    b, counts = np.unique(labels, return_counts=True)
    # dominants = pallets[np.argmax(counts)]

    pallets = np.sort(pallets, -2)
    return pallets.astype(np.uint8), counts


def diff_dominant_color(pallets1, counts1, pallets2, counts2):
    diff_pallets = np.sum((pallets1 - pallets2) ** 2, -1)
    diff_counts = np.abs(counts1 - counts2)
    diff = diff_pallets * diff_counts
    return np.sqrt(np.sum(diff))
