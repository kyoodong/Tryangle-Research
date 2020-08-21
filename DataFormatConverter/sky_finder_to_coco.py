import json
import numpy as np
import cv2
from coco.coco import *
import pycocotools.mask as mask
from glob import glob

with open("datasets/coco_annotations/instances_val2017.json") as coco_json_file:
    json_data = json.load(coco_json_file)
    coco = Coco(json_data)

    labels_info = []
    mask_image_files = glob('datasets/skyfinder/masks/*')

    category_id = coco.add_category('background', 'sky')

    # max_num = 10000
    # for mask_image_file in mask_image_files:
    #     print('process', mask_image_file)
    #     number = mask_image_file.split('/')[-1]
    #     number = number.split('.png')[0]
    #     sky_image_files = glob('datasets/skyfinder/images/{}000*.jpg'.format(number))
    #
    #     added_image_ids = list()
    #     count = 0
    #     for sky_image_file in sky_image_files:
    #         filename = sky_image_file.split('/')[-1]
    #         im = cv2.imread(sky_image_file)
    #         if im is None:
    #             print('im is None', sky_image_file)
    #             continue
    #
    #         id = int(filename.split('.')[0])
    #         if coco.is_exist_image_id(id):
    #             print('{} is already used'.format(id))
    #             continue
    #
    #         coco.add_image(1, filename, None, im.shape[0], im.shape[1], None, None, id)
    #         added_image_ids.append(id)
    #
    #         count += 1
    #         if count == max_num:
    #             break
    #
    #     mask_image = cv2.imread(mask_image_file)
    #     mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    #     _, mask_image = cv2.threshold(mask_image, 127, 255, 0)
    #
    #     contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     segmentation = []
    #     for contour in contours:
    #         contour = contour.flatten().tolist()
    #         # segmentation.append(contour)
    #         if len(contour) > 4:
    #             segmentation.append(contour)
    #     if len(segmentation) == 0:
    #         continue
    #
    #     rle = mask.encode(np.asfortranarray(mask_image))
    #     area = mask.area(rle)
    #     bbox = mask.toBbox(rle)
    #
    #     for added_image_id in added_image_ids:
    #         coco.add_annotation(segmentation, area, 0, added_image_id, [int(v) for v in bbox], category_id)

    print('작성 중...')
    f = open("val_annotation.json", "w")
    f.write(str(coco))
    f.close()
    print('완료!')

