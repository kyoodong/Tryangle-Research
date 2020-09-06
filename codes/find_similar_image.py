import os
import cv2
import skimage.io
import matplotlib.pyplot as plt
import process.color as color_guide
import numpy as np
from process.segmentation import MaskRCNN
from process.color import Color


ROOT_DIR = os.path.abspath("../")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

file_names = next(os.walk(IMAGE_DIR))[2]
COLOR_THRESHOLD = 5000

mask_rcnn = MaskRCNN()

while True:
    # Image1
    image_file_name = input("파일명을 입력하세요 : ")
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, "{}.jpg".format(image_file_name)))
    dominant_color1 = color_guide.find_dominant_colors(image)

    # Image2
    image_file_name2 = input("파일명을 입력하세요 : ")
    image2 = skimage.io.imread(os.path.join(IMAGE_DIR, "{}.jpg".format(image_file_name2)))
    dominant_color2 = color_guide.find_dominant_colors(image2)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()

    print([Color.Category.NAMES[i] for i in dominant_color1], [Color.Category.NAMES[i] for i in dominant_color2])
    if len(np.intersect1d(dominant_color1, dominant_color2)) >= 2:
        print("두 이미지는 비슷한 색을 지녔습니다")
    else:
        print("두 이미지의 색은 비슷하지 않습니다.")

    ### 밝기 보정 소스
    gamma = 0.4
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)

    ### CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 현재까지는 clahe 가 가장 보기 좋음
    # 적어도 effective line 을 찾기에는 유용함
    image = clahe_image

    ### CLAHE
    lab = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 현재까지는 clahe 가 가장 보기 좋음
    # 적어도 effective line 을 찾기에는 유용함
    image2 = clahe_image

    # Run detection
    results = mask_rcnn.detect(image)
    results2 = mask_rcnn.detect(image2)

    main_obj = None


