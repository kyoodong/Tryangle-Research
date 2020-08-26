import os
import skimage.io
import matplotlib.pyplot as plt
import process.guide_image as guide_image
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

while True:
    image_file_name = input("파일명을 입력하세요 : ")
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, "{}.jpg".format(image_file_name)))

    dominant_colors, counts = guide_image.find_dominant_colors(image)
    dominant_color_image = np.zeros((0, 150, 3), np.uint8)
    for dominant_color in dominant_colors:
        im = np.zeros((30 * len(dominant_colors), 150, 3), np.uint8)
        im += dominant_color
        dominant_color_image = np.vstack([dominant_color_image, im])

    plt.imshow(dominant_color_image)
    plt.show()

    color_diff = guide_image.diff_dominant_color(dominant_colors, counts, dominant_colors, counts)
    print('color diff = ', color_diff)
