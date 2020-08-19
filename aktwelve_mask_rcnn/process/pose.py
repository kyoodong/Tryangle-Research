import cv2
import matplotlib.pyplot as plt
import os
from enum import Enum
from process.object import Human
import numpy as np


class HumanPose(Enum):
    Unknown = 0
    Stand = 1
    Sit = 2


# caffemodel 파일 다운로드
# wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel -P pose


class CVPoseEstimator:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        self.net = cv2.dnn.readNetFromCaffe("{}/pose/body_25/pose_deploy.prototxt".format(dirname),
                                       "{}/pose/body_25/pose_iter_584000.caffemodel".format(dirname))

        self.threshold = 0.4

    def inference(self, image):
        height, width = image.shape[0], image.shape[1]
        inp = cv2.dnn.blobFromImage(image, 1.0 / 255, (width, height), (0, 0, 0), False, False)
        self.net.setInput(inp)

        out = self.net.forward()
        points = list()

        for i in range(len(Human.BODY_PARTS)):
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
        #     assert (partFrom in Human.BODY_PARTS)
        #     assert (partTo in Human.BODY_PARTS)
        #
        #     idFrom = Human.BODY_PARTS[partFrom]
        #     idTo = Human.BODY_PARTS[partTo]
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
        if human[Human.BODY_PARTS["LAnkle"]]:
            self.left_ankle = human[Human.BODY_PARTS["LAnkle"]][1]

        if human[Human.BODY_PARTS["RAnkle"]]:
            self.right_ankle = human[Human.BODY_PARTS["RAnkle"]][1]

        if human[Human.BODY_PARTS["LHip"]]:
            self.left_hip = human[Human.BODY_PARTS["LHip"]][1]

        if human[Human.BODY_PARTS["RHip"]]:
            self.right_hip = human[Human.BODY_PARTS["RHip"]][1]

        if human[Human.BODY_PARTS["LKnee"]]:
            self.left_knee = human[Human.BODY_PARTS["LKnee"]][1]

        if human[Human.BODY_PARTS["RKnee"]]:
            self.right_knee = human[Human.BODY_PARTS["RKnee"]][1]

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
        if human.pose[Human.BODY_PARTS[Human.Part.LEar]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.REar]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.LEye]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.REye]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.Nose]] is not None:
                return True
        return False

    def has_upper_body(self, human):
        if human.pose[Human.BODY_PARTS[Human.Part.LShoulder]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.RShoulder]] is not None:
                return True
        return False

    def has_lower_body(self, human):
        if human.pose[Human.BODY_PARTS[Human.Part.LHip]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.RHip]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.MidHip]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.RKnee]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.LKnee]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.RAnkle]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.LAnkle]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.LHeel]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.RHeel]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.LSmallToe]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.RSmallToe]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.LBigToe]] is not None or \
                human.pose[Human.BODY_PARTS[Human.Part.RBigToe]] is not None:
                return True
        return False

    def has_body(self, human):
        return self.has_upper_body(human) or self.has_lower_body(human)

    def run(self, human, image):
        height, width = image.shape[0], image.shape[1]

        for key in Human.BODY_PARTS.keys():
            if human.pose[Human.BODY_PARTS[key]] is not None:
                center = np.array(human.pose[Human.BODY_PARTS[key]]) + np.array([human.roi[1], human.roi[0]])
                center = tuple(center)
                cv2.circle(image, center, 3, (255, 0, 0), thickness=3)
                cv2.putText(image, key, center, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 2)

        plt.imshow(image)
        plt.show()

        # 서 있는 경우
        if human.pose_class == HumanPose.Stand:
            gamma = 5

            # 사람이 사진 밑쪽에 위치한 경우
            if human.roi[2] + gamma > height:
                # 발목이 잘린 경우
                if human.pose[Human.BODY_PARTS["LAnkle"]] is None and human.pose[Human.BODY_PARTS["RAnkle"]] is None\
                        and human.pose[Human.BODY_PARTS["LBigToe"]] is None and human.pose[Human.BODY_PARTS["LSmallToe"]] is None\
                        and human.pose[Human.BODY_PARTS["RBigToe"]] is None and human.pose[Human.BODY_PARTS["RSmallToe"]] is None:
                    return "관절(발목)이 잘리지 않게 발끝에 맞춰 찍어보세요"

                # 무릎이 잘린 경우
                if human.pose[Human.BODY_PARTS["LKnee"]] is None or human.pose[Human.BODY_PARTS["RKnee"]] is None:
                    return "관절(무릎)이 잘리지 않게 허벅지에서 잘라보세요"

                if self.has_head(human):
                    if not self.has_body(human):
                        return "관절(목)이 잘리지 않게 어깨 잘라보세요"

            if height > human.roi[2] + self.foot_lower_threshold:
                return "발 끝을 사진 맨 밑에 맞추세요"

        return None