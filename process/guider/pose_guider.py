from process.object import Human
from process.pose import HumanPose
from process.guide import ObjectGuide


class PoseGuider:
    def __init__(self, human_component):
        self.human_component = human_component
        self.human = self.human_component.object
        self.foot_lower_threshold = 10

    def has_head(self):
        if self.human.pose[Human.BODY_PARTS[Human.Part.LEar]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.REar]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.LEye]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.REye]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.Nose]][2] > HumanPose.POSE_THRESHOLD:
                return True
        return False

    def has_upper_body(self):
        if self.human.pose[Human.BODY_PARTS[Human.Part.LShoulder]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.RShoulder]][2] > HumanPose.POSE_THRESHOLD:
                return True
        return False

    def has_lower_body(self):
        if self.human.pose[Human.BODY_PARTS[Human.Part.LHip]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.RHip]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.RKnee]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.LKnee]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.RAnkle]][2] > HumanPose.POSE_THRESHOLD or \
                self.human.pose[Human.BODY_PARTS[Human.Part.LAnkle]][2] > HumanPose.POSE_THRESHOLD:
                return True
        return False

    def has_body(self):
        return self.has_upper_body() and self.has_lower_body()

    def run(self, image):
        guide_message_list = list()
        height, width = image.shape[0], image.shape[1]

        # 서 있는 경우
        if self.human.pose_class == HumanPose.Stand:
            gamma = 5

            # 사람이 사진 밑쪽에 위치한 경우
            if self.human.roi[2] + gamma > height:
                # Ankle, knee 모두 존재하는 경우 발 끝을 맞춘 것임
                if self.human.pose[Human.BODY_PARTS[Human.Part.LAnkle]][2] > HumanPose.POSE_THRESHOLD and\
                    self.human.pose[Human.BODY_PARTS[Human.Part.RAnkle]][2] > HumanPose.POSE_THRESHOLD and\
                        self.human.pose[Human.BODY_PARTS[Human.Part.LKnee]][2] > HumanPose.POSE_THRESHOLD and\
                        self.human.pose[Human.BODY_PARTS[Human.Part.RKnee]][2] > HumanPose.POSE_THRESHOLD:
                    human_height = self.human.roi[2] - self.human.roi[0]
                    diff = -human_height * 10 / 170
                    guide_message_list.append(ObjectGuide(self.human_component.id, 2, diff, 0))

                # 상반신 사진인 경우
                if self.has_head():
                    # 하체는 없고 상체만 있음 = 상반신 사진
                    if self.has_upper_body() and self.has_lower_body():
                        human_height = self.human.roi[2] - self.human.roi[0]
                        diff = -human_height * 20 / 170
                        guide_message_list.append(
                            ObjectGuide(self.human_component.id, 8, diff, 0))

        return guide_message_list