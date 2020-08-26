import os
import skimage.io
import matplotlib.pyplot as plt
import process.guide_image as guide_image
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
COLOR_THRESHOLD = 5000

while True:
    # Image1
    image_file_name = input("파일명을 입력하세요 : ")
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, "{}.jpg".format(image_file_name)))

    dominant_colors, counts = guide_image.find_dominant_colors(image)
    dominant_color_image = np.zeros((0, 150, 3), np.uint8)
    for dominant_color in dominant_colors:
        im = np.zeros((30 * len(dominant_colors), 150, 3), np.uint8)
        im += dominant_color
        dominant_color_image = np.vstack([dominant_color_image, im])

    # Image2
    image_file_name2 = input("파일명을 입력하세요 : ")
    image2 = skimage.io.imread(os.path.join(IMAGE_DIR, "{}.jpg".format(image_file_name2)))

    dominant_colors2, counts2 = guide_image.find_dominant_colors(image2)
    dominant_color_image2 = np.zeros((0, 150, 3), np.uint8)
    for dominant_color in dominant_colors2:
        im = np.zeros((30 * len(dominant_colors), 150, 3), np.uint8)
        im += dominant_color
        dominant_color_image2 = np.vstack([dominant_color_image2, im])

    plt.subplot(1, 2, 1)
    plt.imshow(dominant_color_image)
    plt.subplot(1, 2, 2)
    plt.imshow(dominant_color_image2)
    plt.show()

    color_diff = guide_image.diff_dominant_color(dominant_colors, counts, dominant_colors2, counts2)
    print('color diff = ', color_diff)

    if color_diff < COLOR_THRESHOLD:
        print("두 이미지는 비슷한 색을 지녔습니다")
    else:
        print("두 이미지는 비슷하지 않습니다.")

