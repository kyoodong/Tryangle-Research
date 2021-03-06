import os
import sys

import skimage.io

from process.component import ObjectComponent
from process.guider.guider import SimpleGuider, ComplexGuider

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


while True:
    image_file_name = input("파일명을 입력하세요 : ")
    image = skimage.io.imread(os.path.join(IMAGE_DIR, "{}.jpg".format(image_file_name)))

    # r = api.segment(image)
    simple_guider = SimpleGuider()
    complex_guider = ComplexGuider()

    simple_guider.guide(image, False)
    complex_guider.guide(image, False)

