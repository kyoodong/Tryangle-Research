import cv2
import matplotlib.pyplot as plt
import os
from tf_pose import common
from enum import Enum


class HumanPose(Enum):
    Unknown = 0
    Stand = 1
    Sit = 2


# caffemodel 파일 다운로드
# wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel -P pose

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]


class CVPoseEstimator:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        self.net = cv2.dnn.readNetFromCaffe("{}/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt".format(dirname),
                                       "{}/pose/mpi/pose_iter_160000.caffemodel".format(dirname))

        self.threshold = 0.4

    def inference(self, image):
        height, width = image.shape[0], image.shape[1]
        inp = cv2.dnn.blobFromImage(image, 1.0 / 255, (width, height), (0, 0, 0), False, False)
        self.net.setInput(inp)

        out = self.net.forward()
        points = list()

        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (width * point[0]) / out.shape[3]
            y = (height * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > self.threshold else None)

        # for pair in POSE_PAIRS:
        #     partFrom = pair[0]
        #     partTo = pair[1]
        #     assert (partFrom in BODY_PARTS)
        #     assert (partTo in BODY_PARTS)
        #
        #     idFrom = BODY_PARTS[partFrom]
        #     idTo = BODY_PARTS[partTo]
        #     if points[idFrom] and points[idTo]:
        #         cv2.line(image, points[idFrom], points[idTo], (255, 74, 0), 3)
        #         cv2.ellipse(image, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
        #         cv2.ellipse(image, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
        #         cv2.putText(image, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
        #                     cv2.LINE_AA)
        #         cv2.putText(image, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
        #                     cv2.LINE_AA)
        return points


class PoseClassifier:
    def run(self, human):
        pass


class TfPoseClassifier(PoseClassifier):
    def run(self, human):
        left_ankle = -1
        right_ankle = -1
        left_hip = -1
        right_hip = -1
        left_knee = -1
        right_knee = -1
        gamma = 0.05

        if common.CocoPart.LAnkle.value in human.body_parts.keys():
            left_ankle = human.body_parts[common.CocoPart.LAnkle.value].y

        if common.CocoPart.RAnkle.value in human.body_parts.keys():
            right_ankle = human.body_parts[common.CocoPart.RAnkle.value].y

        if common.CocoPart.LHip.value in human.body_parts.keys():
            left_hip = human.body_parts[common.CocoPart.LHip.value].y

        if common.CocoPart.RHip.value in human.body_parts.keys():
            right_hip = human.body_parts[common.CocoPart.RHip.value].y

        if common.CocoPart.LKnee.value in human.body_parts.keys():
            left_knee = human.body_parts[common.CocoPart.LKnee.value].y

        if common.CocoPart.RKnee.value in human.body_parts.keys():
            right_knee = human.body_parts[common.CocoPart.RKnee.value].y

        # 무릎의 높이가 엉덩이의 높이보다 낮은 경우, 서 있다(stand)고 판단
        if (left_knee != -1 and left_hip != -1 and left_knee > left_hip + gamma) or \
                (right_knee != -1 and right_hip != -1 and right_knee > right_hip + gamma):
            return HumanPose.Stand

        return HumanPose.Unknown


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

        if human[BODY_PARTS["LAnkle"]]:
            self.left_ankle = human[BODY_PARTS["LAnkle"]][1]

        if human[BODY_PARTS["RAnkle"]]:
            self.right_ankle = human[BODY_PARTS["RAnkle"]][1]

        if human[BODY_PARTS["LHip"]]:
            self.left_hip = human[BODY_PARTS["LHip"]][1]

        if human[BODY_PARTS["RHip"]]:
            self.right_hip = human[BODY_PARTS["RHip"]][1]

        if human[BODY_PARTS["LKnee"]]:
            self.left_knee = human[BODY_PARTS["LKnee"]][1]

        if human[BODY_PARTS["RKnee"]]:
            self.right_knee = human[BODY_PARTS["RKnee"]][1]

    def run(self, human):
        self.__extract(human)

        # 무릎의 높이가 엉덩이의 높이보다 낮은 경우, 서 있다(stand)고 판단
        if (self.left_knee != -1 and self.left_hip != -1 and self.left_knee > self.left_hip + self.gamma) or \
                (self.right_knee != -1 and self.right_hip != -1 and self.right_knee > self.right_hip + self.gamma):
            return HumanPose.Stand

        return HumanPose.Unknown


class PoseGuider:
    def __init__(self):
        self.foot_lower_threshold = 10

    def run(self, cropped_image, human, human_pose, image, roi):
        height, width = image.shape[0], image.shape[1]
        cropped_height, cropped_width = cropped_image.shape[0], cropped_image.shape[1]

        if human_pose == HumanPose.Stand:
            # 서 있지만 발목이 잘린 경우
            if human[BODY_PARTS["LAnkle"]] is None or human[BODY_PARTS["RAnkle"]] is None:
                return "사람 발목이 잘리지 않게 하세요"

            if height > roi[2] + self.foot_lower_threshold:
                return "발 끝을 사진 맨 밑에 맞추세요"

        return None