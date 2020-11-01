from process.object import Human
from process.pose import HumanPose


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
        return self.has_upper_body() or self.has_lower_body()

    def run(self, image):
        guide_message_list = list()
        height, width = image.shape[0], image.shape[1]

        # 서 있는 경우
        if self.human.pose_class == HumanPose.Stand:
            gamma = 5

            # 사람이 사진 밑쪽에 위치한 경우
            if self.human.roi[2] + gamma > height:
                # 발목이 잘린 경우
                # 무릎은 있으나 발목, 발꿈치 등이 모두 없는 경우
                if self.human.pose[Human.BODY_PARTS[Human.Part.LAnkle]][2] <= HumanPose.POSE_THRESHOLD and\
                    self.human.pose[Human.BODY_PARTS[Human.Part.RAnkle]][2] <= HumanPose.POSE_THRESHOLD and\
                        self.human.pose[Human.BODY_PARTS[Human.Part.LKnee]][2] > HumanPose.POSE_THRESHOLD and\
                        self.human.pose[Human.BODY_PARTS[Human.Part.RKnee]][2] > HumanPose.POSE_THRESHOLD:
                    human_height = self.human.roi[2] - self.human.roi[0]
                    diff = -human_height * 10 / 170
                    guide_message_list.append(ObjectGuide(self.human_component.id, 3, diff, 0))

                # 무릎이 잘린 경우
                if self.human.pose[Human.BODY_PARTS[Human.Part.LKnee]][2] <= HumanPose.POSE_THRESHOLD or\
                        self.human.pose[Human.BODY_PARTS[Human.Part.RKnee]][2] <= HumanPose.POSE_THRESHOLD:
                    human_height = self.human.roi[2] - self.human.roi[0]
                    diff = human_height * 20 / 170
                    guide_message_list.append(
                        ObjectGuide(self.human_component.id, 7, diff, 0))

                if self.has_head():
                    if not self.has_body():
                        human_height = self.human.roi[2] - self.human.roi[0]
                        diff = -human_height * 20 / 170
                        guide_message_list.append(
                            ObjectGuide(self.human_component.id, 8, diff, 0))

            if height > self.human.roi[2] + self.foot_lower_threshold:
                guide_message_list.append(
                    ObjectGuide(self.human_component.id, 2, height - self.human.roi[2] + self.foot_lower_threshold, 0))

            # 사람이 사진의 윗쪽에 위치한 경우
            if self.human.roi[0] < gamma:
                # 머리가 있다면
                if self.has_head():
                    top = height // 3
                    diff = top - self.human.roi[0]
                    guide_message_list.append(
                        ObjectGuide(self.human_component.id, 9, diff, 0))

        return guide_message_list