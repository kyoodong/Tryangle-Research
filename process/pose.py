from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
# import models 필수!
import models
from config import cfg
from config import check_config
from config import update_config
from core.group import HeatmapParser
from core.inference import aggregate_results
from core.inference import get_multi_stage_outputs
from fp16_utils.fp16util import network_to_half
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import resize_align_multi_scale
from utils.utils import create_logger
from utils.utils import get_model_summary

from process.object import Human

torch.multiprocessing.set_sharing_strategy('file_system')


class HumanPose:
    Unknown = 0
    Stand = 1
    Sit = 2

    POSE_THRESHOLD = 0.02


# caffemodel 파일 다운로드
# wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel -P pose

class Arg:
    def __init__(self):
        self.cfg = None
        self.opts = None


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

args = Arg()
args.cfg = '{}/human_pose/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml'.format(ROOT_DIR)
args.opts = list()
args.opts.append('TEST.MODEL_FILE')
args.opts.append('{}/human_pose/models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth'.format(ROOT_DIR))

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
        if human[Human.BODY_PARTS[Human.Part.LHip]][2] > HumanPose.POSE_THRESHOLD:
            self.left_hip = human[Human.BODY_PARTS[Human.Part.LHip]][1]

        if human[Human.BODY_PARTS[Human.Part.RHip]][2] > HumanPose.POSE_THRESHOLD:
            self.right_hip = human[Human.BODY_PARTS[Human.Part.RHip]][1]

        if human[Human.BODY_PARTS[Human.Part.LKnee]][2] > HumanPose.POSE_THRESHOLD:
            self.left_knee = human[Human.BODY_PARTS[Human.Part.LKnee]][1]

        if human[Human.BODY_PARTS[Human.Part.RKnee]][2] > HumanPose.POSE_THRESHOLD:
            self.right_knee = human[Human.BODY_PARTS[Human.Part.RKnee]][1]

    def run(self, pose):
        self.__extract(pose)

        # 무릎의 높이가 엉덩이의 높이보다 낮은 경우, 서 있다(stand)고 판단
        if (self.left_knee != -1 and self.left_hip != -1 and self.left_knee > self.left_hip + self.gamma) or \
                (self.right_knee != -1 and self.right_hip != -1 and self.right_knee > self.right_hip + self.gamma):
            return HumanPose.Stand

        return HumanPose.Unknown