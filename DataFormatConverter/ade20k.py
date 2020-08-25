import json
import numpy as np
import cv2
from coco.coco import *
import pycocotools.mask as mask
from glob import glob
import matplotlib.pyplot as plt
import random
import os

SUPER_CLASS = "background"
CLASS_NAME = "ground"
DISPLAY_CLASS_NAMES = ["ground", "grass"]
LOWER_CLASS_COLORS = [(0, 70, 30), (0, 101, 40)]
UPPER_CLASS_COLORS = [(256, 70, 30), (256, 101, 40)]

# SUPER_CLASS = "background"
# CLASS_NAME = "sea"
# DISPLAY_CLASS_NAMES = ["sea"]
# LOWER_CLASS_COLORS = [(0, 216, 80)]
# UPPER_CLASS_COLORS = [(130, 216, 80)]

# SUPER_CLASS = "background"
# CLASS_NAME = "ground"
# DISPLAY_CLASS_NAMES = ["sky"]
# LOWER_CLASS_COLORS = [(0, 116, 90)]
# UPPER_CLASS_COLORS = [(80, 116, 90)]

MODE = "train"
# MODE = "val"

COCO_IMAGES_DIR_PATH = 'datasets/ADE20K_2016_07_26/coco_{}_images'.format(MODE)

if not os.path.exists(COCO_IMAGES_DIR_PATH):
    os.mkdir(COCO_IMAGES_DIR_PATH)

with open("datasets/coco_annotations/instances_{}2017.json".format(MODE)) as coco_json_file:
    json_data = json.load(coco_json_file)
    coco = Coco(json_data)

    labels_info = []
    DIR = "training"
    if MODE == "train":
        DIR = "training"
    else:
        DIR = "validation"
    mask_image_file_paths = glob('datasets/ADE20K_2016_07_26/images/{}/*/*/*_seg.png'.format(DIR))
    # mask_image_file_paths.sort()

    category_id = coco.get_category_id(CLASS_NAME)
    if category_id == -1:
        category_id = coco.add_category(SUPER_CLASS, CLASS_NAME)

    for mask_image_file_path in mask_image_file_paths:
        dir_list = mask_image_file_path.split('/')
        alphabet = dir_list[-3]
        word = dir_list[-2]
        filename_prefix = dir_list[-1]
        filename_prefix = filename_prefix.split('_seg.png')[0]
        origin_image_file_path = 'datasets/ADE20K_2016_07_26/images/{}/{}/{}/{}.jpg'.format(
            DIR, alphabet, word, filename_prefix)
        attribute_file_path = 'datasets/ADE20K_2016_07_26/images/{}/{}/{}/{}_atr.txt'.format(
            DIR, alphabet, word, filename_prefix)

        has_class = False
        with open(attribute_file_path) as attribute_file:
            while True:
                line_str = attribute_file.readline()
                if not line_str:
                    break

                data_list = line_str.split(" # ")
                for display_class_name in DISPLAY_CLASS_NAMES:
                    if data_list[3] == display_class_name or data_list[4] == display_class_name:
                        has_class = True
                        break

        if not has_class:
            continue

        filename = origin_image_file_path.split('/')[-1]
        im = cv2.imread(origin_image_file_path)

        if im is None:
            print("im is None", origin_image_file_path)
            continue

        origin_mask_image = cv2.imread(mask_image_file_path)
        # hsv = cv2.cvtColor(origin_mask_image, cv2.COLOR_BGR2HSV)
        mask_all = np.zeros([origin_mask_image.shape[0], origin_mask_image.shape[1]])
        for i in range(len(LOWER_CLASS_COLORS)):
            img_mask = cv2.inRange(origin_mask_image, LOWER_CLASS_COLORS[i], UPPER_CLASS_COLORS[i])
            mask_all = np.clip(mask_all + img_mask, 0, 255)
        mask_all = mask_all.astype('uint8')
        img_result = cv2.bitwise_and(origin_mask_image, origin_mask_image, mask=mask_all)
        mask_image = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
        _, mask_image = cv2.threshold(mask_image, 1, 255, 0)

        # plt.subplot(2, 2, 1)
        # plt.title(filename_prefix)
        # plt.imshow(im)
        # plt.subplot(2, 2, 2)
        # plt.imshow(origin_mask_image)
        # plt.subplot(2, 2, 3)
        # plt.imshow(img_result)
        # plt.subplot(2, 2, 4)
        # plt.imshow(mask_image, 'gray')
        # plt.show()

        contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                segmentation.append(contour)
        if len(segmentation) == 0:
            continue

        rle = mask.encode(np.asfortranarray(mask_image))
        area = mask.area(rle)
        bbox = mask.toBbox(rle)

        id = coco.find_image_id_by_filename(filename)

        # 해당 파일을 갖고 있지 않으면 coco에 이미지 파일 추가
        if id == -1:
            id = coco.get_new_image_id()
            coco.add_image(1, filename, None, im.shape[0], im.shape[1], None, None, id)
            print('{} is created'.format(filename))
        else:
            print('{} is already exist'.format(filename))

        print('add annotation', origin_image_file_path)
        coco.add_annotation(segmentation, area, 0, id, [int(v) for v in bbox], category_id)

        dst_path = "{}/{}".format(COCO_IMAGES_DIR_PATH, filename)
        if not os.path.exists(dst_path):
            op = "cp {} {}".format(origin_image_file_path, dst_path)
            os.system(op)

    print('작성 중...')
    f = open("{}_annotation.json".format(MODE), "w")
    f.write(str(coco))
    f.close()
    print('완료!')

