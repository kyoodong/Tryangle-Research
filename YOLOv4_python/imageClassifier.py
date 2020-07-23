import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from core.config import cfg
import cv2
from PIL import Image
import numpy as np
import os
import glob


class Classifier:

    def __init__(self):

        self.name_dic = utils.read_class_names(cfg.YOLO.CLASSES)
        self.input_size = 416

    def load_image(self, path):
        images_data = []
        if os.path.isdir(path):
            tmp = [cv2.imread(file) for file in glob.glob(path + "/*.jpg")]
            for org_img in tmp:
                org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

                image_data = cv2.resize(org_img, (self.input_size, self.input_size))
                image_data = image_data / 255.
                images_data.append(image_data)
        else:
            original_image = cv2.imread(path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (self.input_size, self.input_size))
            image_data = image_data / 255.
            images_data.append(image_data)

        images_data = np.asarray(images_data).astype(np.float32)
        return images_data

    def get_object_list(self, input, weight='./data/yolov4-416', iou=0.45, score=0.25):
        '''STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()'''

        batch_data = tf.constant(self.load_image(input))

        # Model을 로드하는 코드, 시간 소모가 큼
        saved_model_loaded = tf.saved_model.load(weight, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        # 예측하는 부분
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        '''
        pred_bbox 을 이용해서 유효한 정보를 뽑아냄
        
        tf.image.combined_non_max_suppression
        returns: 
            'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor containing the non-max suppressed boxes. 
            'nmsed_scores': A [batch_size, max_detections] float32 tensor containing the scores for the boxes. 
            'nmsed_classes': A [batch_size, max_detections] float32 tensor containing the class for boxes. 
            'valid_detections': A [batch_size] int32 tensor indicating the number of valid detections per batch item. 
            Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. 
            The rest of the entries are zero paddings.
        '''
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        find_list = []
        for i in range(len(classes.numpy())):
            find_list.append([])
            for t in range(valid_detections[i]):
                find_list[i].append(self.name_dic[classes.numpy()[i, t]])
        return find_list