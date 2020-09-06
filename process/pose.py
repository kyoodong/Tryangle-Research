from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import os
from enum import Enum
from process.object import Human

import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import skimage
import cv2
import numpy as np
import copy

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')


class HumanPose(Enum):
    Unknown = 0
    Stand = 1
    Sit = 2


# caffemodel 파일 다운로드
# wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel -P pose

class Arg:
    def __init__(self):
        self.cfg = None
        self.opts = None


args = Arg()
args.cfg = '../human_pose/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml'
args.opts = list()
args.opts.append('TEST.MODEL_FILE')
args.opts.append('../human_pose/models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth')

update_config(cfg, args)
check_config(cfg)

logger, final_output_dir, tb_log_dir = create_logger(
    cfg, args.cfg, 'valid'
)

logger.info(pprint.pformat(args))
logger.info(cfg)

# cudnn related setting
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
    cfg, is_train=False
)

dump_input = torch.rand(
    (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
)
logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

if cfg.FP16.ENABLED:
    model = network_to_half(model)

if cfg.TEST.MODEL_FILE:
    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=True)
else:
    model_state_file = os.path.join(
        final_output_dir, 'model_best.pth.tar'
    )
    logger.info('=> loading model from {}'.format(model_state_file))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_state_file))
    else:
        model.load_state_dict(torch.load(model_state_file, map_location=torch.device('cpu')))

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model).cpu()
model.eval()

if cfg.MODEL.NAME == 'pose_hourglass':
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
else:
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

parser = HeatmapParser(cfg)
POSE_THRESHOLD = 0.02


class CVPoseEstimator:
    def inference(self, image):
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                if torch.cuda.is_available():
                    image_resized = image_resized.unsqueeze(0).cuda()
                else:
                    image_resized = image_resized.unsqueeze(0).cpu()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

            if len(scores) == 0:
                return None

            max_index = np.argmax(scores)

            # Visualization
            # result_image = copy.copy(image)
            # for key in Human.BODY_PARTS.keys():
            #     result = final_results[max_index][Human.BODY_PARTS[key]]
            #     if result[2] > POSE_THRESHOLD:
            #         cv2.circle(result_image, (result[0], result[1]), 4, (255, 0, 0))
            #         cv2.putText(result_image, key, (result[0], result[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255),
            #                     1)
            #
            # plt.subplot(1, 2, 1)
            # plt.imshow(image)
            # plt.subplot(1, 2, 2)
            # plt.imshow(result_image)
            # plt.show()
            return final_results[max_index][:, :3]


class PoseClassifier:
    def run(self, human):
        pass


class CvClassifier(PoseClassifier):
    def __init__(self):
        self.gamma = 0.05

    def __extract(self, human):
        self.left_ankle = -1
        self.right_ankle = -1
        self.left_hip = -1
        self.right_hip = -1
        self.left_knee = -1
        self.right_knee = -1

        # pose estimation 결과 중 유의미한 정보만을 뽑아냄
        if human[Human.BODY_PARTS[Human.Part.LHip]][2] > POSE_THRESHOLD:
            self.left_hip = human[Human.BODY_PARTS[Human.Part.LHip]][1]

        if human[Human.BODY_PARTS[Human.Part.RHip]][2] > POSE_THRESHOLD:
            self.right_hip = human[Human.BODY_PARTS[Human.Part.RHip]][1]

        if human[Human.BODY_PARTS[Human.Part.LKnee]][2] > POSE_THRESHOLD:
            self.left_knee = human[Human.BODY_PARTS[Human.Part.LKnee]][1]

        if human[Human.BODY_PARTS[Human.Part.RKnee]][2] > POSE_THRESHOLD:
            self.right_knee = human[Human.BODY_PARTS[Human.Part.RKnee]][1]

    def run(self, pose):
        self.__extract(pose)

        # 무릎의 높이가 엉덩이의 높이보다 낮은 경우, 서 있다(stand)고 판단
        if (self.left_knee != -1 and self.left_hip != -1 and self.left_knee > self.left_hip + self.gamma) or \
                (self.right_knee != -1 and self.right_hip != -1 and self.right_knee > self.right_hip + self.gamma):
            return HumanPose.Stand

        return HumanPose.Unknown


class PoseGuider:
    def __init__(self):
        self.foot_lower_threshold = 10

    def has_head(self, human):
        if human.pose[Human.BODY_PARTS[Human.Part.LEar]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.REar]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.LEye]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.REye]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.Nose]][2] > POSE_THRESHOLD:
                return True
        return False

    def has_upper_body(self, human):
        if human.pose[Human.BODY_PARTS[Human.Part.LShoulder]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.RShoulder]][2] > POSE_THRESHOLD:
                return True
        return False

    def has_lower_body(self, human):
        if human.pose[Human.BODY_PARTS[Human.Part.LHip]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.RHip]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.RKnee]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.LKnee]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.RAnkle]][2] > POSE_THRESHOLD or \
                human.pose[Human.BODY_PARTS[Human.Part.LAnkle]][2] > POSE_THRESHOLD:
                return True
        return False

    def has_body(self, human):
        return self.has_upper_body(human) or self.has_lower_body(human)

    def run(self, human, image):
        guide_message_list = list()
        height, width = image.shape[0], image.shape[1]

        for key in Human.BODY_PARTS.keys():
            if human.pose[Human.BODY_PARTS[key]][2] > POSE_THRESHOLD:
                center = np.array(human.pose[Human.BODY_PARTS[key]][:2]) + np.array([human.extended_roi[1], human.extended_roi[0]])
                center = tuple(center)
                # cv2.circle(image, center, 3, (255, 0, 0), thickness=3)
                # cv2.putText(image, key, center, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 2)

        plt.imshow(image)
        plt.show()

        # 서 있는 경우
        if human.pose_class == HumanPose.Stand:
            gamma = 5

            # 사람이 사진 밑쪽에 위치한 경우
            if human.roi[2] + gamma > height:
                # 발목이 잘린 경우
                # 무릎은 있으나 발목, 발꿈치 등이 모두 없는 경우
                if human.pose[Human.BODY_PARTS[Human.Part.LAnkle]][2] <= POSE_THRESHOLD and human.pose[Human.BODY_PARTS[Human.Part.RAnkle]][2] <= POSE_THRESHOLD\
                        and human.pose[Human.BODY_PARTS[Human.Part.LKnee]][2] > POSE_THRESHOLD and human.pose[Human.BODY_PARTS[Human.Part.RKnee]][2] > POSE_THRESHOLD:
                    human_height = human[2] - human[0]
                    diff = -human_height * 10 / 170
                    guide_message_list.append(("발목이 잘리지 않게 발끝에 맞춰 찍어보세요", (diff, 0)))

                # 무릎이 잘린 경우
                if human.pose[Human.BODY_PARTS["LKnee"]][2] <= POSE_THRESHOLD or human.pose[Human.BODY_PARTS["RKnee"]][2] <= POSE_THRESHOLD:
                    human_height = human[2] - human[0]
                    diff = human_height * 20 / 170
                    guide_message_list.append(("무릎이 잘리지 않게 허벅지까지만 찍어보세요", (diff, 0)))

                if self.has_head(human):
                    if not self.has_body(human):
                        human_height = human[2] - human[0]
                        diff = -human_height * 20 / 170
                        guide_message_list.append(("관절(목)이 잘리지 않게 어깨까지 찍어보세요", (diff, 0)))

            if height > human.roi[2] + self.foot_lower_threshold:
                guide_message_list.append(("발 끝을 사진 맨 밑에 맞추세요", (height - human.roi[2] + self.foot_lower_threshold, 0)))

            # 사람이 사진의 윗쪽에 위치한 경우
            if human.roi[0] < gamma:
                # 머리가 있다면
                if self.has_head(human):
                    top = height // 3
                    diff = top - human.roi[0]
                    guide_message_list.append(("머리 위에는 여백이 있는 것이 좋습니다", (diff, 0)))

        return guide_message_list