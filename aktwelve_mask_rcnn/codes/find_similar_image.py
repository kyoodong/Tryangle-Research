import os
import cv2
import skimage.io
import matplotlib.pyplot as plt
import process.guide_image as guide_image
import numpy as np
from process.segmentation import MaskRCNN


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
    all_layered_image, guided_all_layered_image, sub_obj_list = mask_rcnn.get_layered_image(image, results)
    all_layered_image2, guided_all_layered_image2, sub_obj_list2 = mask_rcnn.get_layered_image(image2, results2)
