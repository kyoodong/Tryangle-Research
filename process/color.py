import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math


# 색상 비교 코드 : https://github.com/sumtype/CIEDE2000
# Converts RGB pixel array to XYZ format.
# Implementation derived from http://www.easyrgb.com/en/math.php
def rgb2xyz(rgb):
    def format(c):
        c = c / 255.
        if c > 0.04045: c = ((c + 0.055) / 1.055) ** 2.4
        else: c = c / 12.92
        return c * 100
    rgb = list(map(format, rgb))
    xyz = [None, None, None]
    xyz[0] = rgb[0] * 0.4124 + rgb[1] * 0.3576 + rgb[2] * 0.1805
    xyz[1] = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
    xyz[2] = rgb[0] * 0.0193 + rgb[1] * 0.1192 + rgb[2] * 0.9505
    return xyz

# Converts XYZ pixel array to LAB format.
# Implementation derived from http://www.easyrgb.com/en/math.php
def xyz2lab(xyz):
    def format(c):
        if c > 0.008856: c = c ** (1. / 3.)
        else: c = (7.787 * c) + (16. / 116.)
        return c
    xyz[0] = xyz[0] / 95.047
    xyz[1] = xyz[1] / 100.00
    xyz[2] = xyz[2] / 108.883
    xyz = list(map(format, xyz))
    lab = [None, None, None]
    lab[0] = (116. * xyz[1]) - 16.
    lab[1] = 500. * (xyz[0] - xyz[1])
    lab[2] = 200. * (xyz[1] - xyz[2])
    return lab

# Converts RGB pixel array into LAB format.
def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

# Returns CIEDE2000 comparison results of two LAB formatted colors.
# Translated from CIEDE2000 implementation in https://github.com/markusn/color-diff
def ciede2000(lab1, lab2):
    def degrees(n): return n * (180. / np.pi)
    def radians(n): return n * (np.pi / 180.)
    def hpf(x, y):
        if x == 0 and y == 0: return 0
        else:
            tmphp = degrees(np.arctan2(x, y))
            if tmphp >= 0: return tmphp
            else: return tmphp + 360.
        return None
    def dhpf(c1, c2, h1p, h2p):
        if c1 * c2 == 0: return 0
        elif np.abs(h2p - h1p) <= 180: return h2p - h1p
        elif h2p - h1p > 180: return (h2p - h1p) - 360.
        elif h2p - h1p < 180: return (h2p - h1p) + 360.
        else: return None
    def ahpf(c1, c2, h1p, h2p):
        if c1 * c2 == 0: return h1p + h2p
        elif np.abs(h1p - h2p) <= 180: return (h1p + h2p) / 2.
        elif np.abs(h1p - h2p) > 180 and h1p + h2p < 360: return (h1p + h2p + 360.) / 2.
        elif np.abs(h1p - h2p) > 180 and h1p + h2p >= 360: return (h1p + h2p - 360.) / 2.
        return None
    L1 = lab1[0]
    A1 = lab1[1]
    B1 = lab1[2]
    L2 = lab2[0]
    A2 = lab2[1]
    B2 = lab2[2]
    kL = 1
    kC = 1
    kH = 1
    C1 = np.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = np.sqrt((A2 ** 2.) + (B2 ** 2.))
    aC1C2 = (C1 + C2) / 2.
    G = 0.5 * (1. - np.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25. ** 7.))))
    a1P = (1. + G) * A1
    a2P = (1. + G) * A2
    c1P = np.sqrt((a1P ** 2.) + (B1 ** 2.))
    c2P = np.sqrt((a2P ** 2.) + (B2 ** 2.))
    h1P = hpf(B1, a1P)
    h2P = hpf(B2, a2P)
    dLP = L2 - L1
    dCP = c2P - c1P
    dhP = dhpf(C1, C2, h1P, h2P)
    dHP = 2. * np.sqrt(c1P * c2P) * np.sin(radians(dhP) / 2.)
    aL = (L1 + L2) / 2.
    aCP = (c1P + c2P) / 2.
    aHP = ahpf(C1, C2, h1P, h2P)
    T = 1. - 0.17 * np.cos(radians(aHP - 39)) + 0.24 * np.cos(radians(2. * aHP)) + 0.32 * np.cos(radians(3. * aHP + 6.)) - 0.2 * np.cos(radians(4. * aHP - 63.))
    dRO = 30. * np.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
    rC = np.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))
    sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / np.sqrt(20. + ((aL - 50.) ** 2.)))
    sC = 1. + 0.045 * aCP
    sH = 1. + 0.015 * aCP * T
    rT = -2. * rC * np.sin(radians(2. * dRO))
    return np.sqrt(((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.) + ((dHP / (sH * kH)) ** 2.) + rT * (dCP / (sC * kC)) * (dHP / (sH * kH)))


def get_similar_color_index(color):
    diff = sys.maxsize
    index = -1
    for i, category_color in enumerate(Color.Category.LIST):
        value = ciede2000(rgb2lab(color), rgb2lab(category_color))
        if value < diff:
            index = i
            diff = value
    return index


class Color:
    class Category:
        NAMES = [
            "Yellow",
            "Yellow_orange",
            "Orange",
            "Red_orange",
            "Red",
            "Pink",
            "Red_violet",
            "Violet",
            "Blue_violet",
            "Blue",
            "Sky",
            "Blue_green",
            "Green",
            "Yellow_green",
            "White",
            "Gray",
            "Brown",
            "Black",
        ]

        LIST = [(246, 239, 30),     # Yellow
                (249, 197, 14),     # Yellow_orange
                (244, 125, 25),     # Orange
                (233, 62, 29),      # Red_orange
                (229, 0, 29),       # Red
                (253, 181, 211),    # Pink
                (139, 41, 134),     # Red_violet
                (100, 27, 128),     # Violet
                (81, 69, 152),      # Blue_violet
                (49, 80, 162),      # Blue
                (135, 198, 228),    # Sky
                (28, 128, 107),     # Blue_green
                (45, 170, 64),      # Green
                (127, 191, 51),     # Yellow_green
                (255, 255, 255),    # White
                (186, 186, 186),    # Gray
                (125, 69, 36),      # Brown
                (0, 0, 0),          # Black
                ]

    def __init__(self, image):
        self.image_size = (32, 32)
        self.dominant_color_num = 3

        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.resize(image, self.image_size)
        visual = np.zeros_like(image)

        self.color_list = list()
        for i in range(len(Color.Category.LIST)):
            self.color_list.append(0)

        for y in range(self.image_size[1]):
            for x in range(self.image_size[1]):
                index = get_similar_color_index(image[y][x])
                self.color_list[index] += 1
                visual[y][x] = Color.Category.LIST[index]

        self.color_list = np.array(self.color_list)

    def __sub__(self, other):
        if isinstance(other, Color):
            return np.log(sum((self.color_list - other.color_list) ** 2))

    def __str__(self):
        string = '{{color_list: {}}}'.format(self.color_list)
        return string

    def get_dominant_colors(self):
        color_list = np.argsort(self.color_list)
        return color_list[-3:]


def find_dominant_colors(image):
    color = Color(image)
    return color.get_dominant_colors()


# pallet = np.zeros([0, 150, 3], dtype=np.int32)
# for i in range(len(Color.Category.LIST)):
#     img = np.zeros([20, 150, 3], dtype=np.int32)
#     img += Color.Category.LIST[i]
#     pallet = np.vstack([pallet, img])
#
# plt.imshow(pallet)
# plt.show()