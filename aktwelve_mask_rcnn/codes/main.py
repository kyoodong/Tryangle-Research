import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from process import image_api, hough
from process.object import Object, Human
import cv2
import copy
from process.pose import CVPoseEstimator, PoseGuider, CvClassifier, HumanPose

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "codes/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO has 80 classes + sky

config = InferenceConfig()
config.display()

# Create model obj in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Last
# model.load_weights(model.find_last(), by_name=True)

# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
#
# # Print class names
# print(dataset.class_names)

#%%

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

#%% md

## Run obj Detection

#%%

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
cv_estimator = CVPoseEstimator()
pose_classifier = CvClassifier()

while True:
    image_file_name = input("파일명을 입력하세요 : ")
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, "{}.jpg".format(image_file_name)))

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

    dst = np.hstack([image, res, clahe_image])
    plt.imshow(dst)
    plt.show()

    # 현재까지는 clahe 가 가장 보기 좋음
    # 적어도 effective line 을 찾기에는 유용함
    image = clahe_image

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    # 객체마다 외곽선만 따도록 수정
    # [image_height][image_width][num_of_obj]
    # 위와 같은 shape 로 이미지가 처리되며 num_of_obj 개수로 나뉜 이미지들을 하나의 이미지로 합쳐야함
    layered_images, center_points = image_api.get_contour_center_point(r['masks'], 0.01)

    obj_list = list()

    for index, center_point in enumerate(center_points):
        if center_point:
            # roi 를 살짝 넓직하게 잡아야 사람 포즈 인식이 잘됨
            roi = r['rois'][index]
            d = 30
            roi[0] -= d
            roi[1] -= d
            roi[2] += d
            roi[3] += d

            height, width = image.shape[0], image.shape[1]
            roi[0] = np.clip(roi[0], 0, width)
            roi[1] = np.clip(roi[1], 0, height)
            roi[2] = np.clip(roi[2], 0, width)
            roi[3] = np.clip(roi[3], 0, height)
            obj = Object(roi, r['masks'][index], r['class_ids'][index], r['scores'][index],
                         center_point)
            if obj.is_person():
                # 사진 내 여러 사람이 있을 수 있으므로 해당 객체만을 오려내서 pose estimation 을 돌림
                cropped_image = image[obj.roi[0]: obj.roi[2], obj.roi[1]:obj.roi[3]]
                plt.imshow(cropped_image)
                plt.show()

                pose = cv_estimator.inference(cropped_image)
                pose_class = pose_classifier.run(pose)
                human = Human(obj, pose, pose_class, cropped_image)
                obj_list.append(human)

            else:
                obj_list.append(obj)

    # all_layered_image 는 레이아웃화 된 객체 이미지들을 하나의 이미지로 합치는 변수
    all_layered_image = np.zeros([image.shape[0], image.shape[1], 1])

    # 합치기
    for i in range(layered_images.shape[-1]):
        all_layered_image[:, :, 0] += layered_images[:, :, i]

    # 객체의 중앙 지점을 파악하여 그에 맞는 가이드를 요청함
    for obj in obj_list:
        all_layered_image = cv2.circle(all_layered_image, obj.center_point, 5, 1, 2)
        guide_message_list = image_api.recommend_obj_position(obj, image)
        print(guide_message_list)

    # 중요한 선을 찾음
    effective_lines = hough.find_hough_line(image)
    if effective_lines:
        for line in effective_lines:
            guide_message_list = image_api.recommend_line_position(line)
            print(guide_message_list)

    # 중요한 선을 시각화하기 위함
    if effective_lines:
        line_image = copy.copy(image)
        for line in effective_lines:
            all_layered_image = cv2.line(all_layered_image, (line[0], line[1]), (line[2], line[3]), 1, 2)
            line_image = cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), 1, 2)

    # 선으로 이루어진 객체들 시각화
    plt.imshow(all_layered_image, 'gray', vmin=0, vmax=1)
    plt.show()

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    if obj_list and effective_lines:
        guide_message_list = image_api.get_guide_message_for_obj_line(obj_list, effective_lines, image)
        print(guide_message_list)

