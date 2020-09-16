from process import text_guider, hough
from process.object import Object, Human
import copy
import process.image_api as api
import numpy as np
from process.pose import CVPoseEstimator, CvClassifier, HumanPose

cv_estimator = CVPoseEstimator()
pose_classifier = CvClassifier()


class Component:
    def __init__(self, id):
        self.id = id


class LineComponent(Component):
    def __init__(self, id, line):
        super(LineComponent, self).__init__(id)
        self.line = line


class ObjectComponent(Component):
    def __init__(self, id, object):
        super(ObjectComponent, self).__init__(id)
        self.object = object


class Guide():
    def __init__(self, object_id, guide_id):
        self.object_id = object_id
        self.guide_id = guide_id

    def __str__(self):
        return "{{'Guide':{}}}".format(str(self.__dict__))

    def __repr__(self):
        return self.__str__()


class LineGuide(Guide):
    def __init__(self, object_id, guide_id):
        super(LineGuide, self).__init__(object_id, guide_id)

    def __str__(self):
        return "{{'LineGuide':{}}}".format(str(self.__dict__))

    def __repr__(self):
        return self.__str__()


class ObjectGuide(Guide):
    def __init__(self, object_id, guide_id, diff_x, diff_y, object_class):
        super(ObjectGuide, self).__init__(object_id, guide_id)
        self.diff_x = diff_x
        self.diff_y = diff_y
        self.object_class = object_class

    def __str__(self):
        return "{{'ObjectGuide':{}}}".format(str(self.__dict__))

    def __repr__(self):
        return self.__str__()


class Guider:
    guide_list = [
        "수평선을 맞추어 찍어 보세요",
        "수직선을 맞추어 찍어 보세요",
        "발 끝을 맞추어 찍으면 다리가 길게 보입니다",
        "발목에서 잘리면 사진이 불안정해 보입니다. 발 끝을 맞추어 찍어 보세요",
        "대상을 중앙에 두어 좌우 대칭을 맞추어 찍어 보세요",
        "대상을 황금 영역에 두고 찍어 보세요",
        "선이 목을 지나는 것보다 머리, 어깨, 허리를 지나면 좋습니다",
        "무릎이 잘리지 않게 허벅지까지만 찍어보세요",
        "목이 잘리지 않게 어깨까지 찍어보세요",
        "머리 위에 여백이 있는것이 좋습니다",
    ]

    def __init__(self, image):
        self.image = image
        self.component_list = list()
        self.guide_list = list()
        for guide in range(len(Guider.guide_list)):
            self.guide_list.append([])

        self.r = api.segment(image)
        self.get_object_and_guide()
        self.get_effective_line_and_guide()

        for component in self.component_list:
            self.get_obj_position_guides(component)

    def get_object_and_guide(self):
        # 객체마다 외곽선만 따도록 수정
        # [image_height][image_width][num_of_obj]
        # 위와 같은 shape 로 이미지가 처리되며 num_of_obj 개수로 나뉜 이미지들을 하나의 이미지로 합쳐야함
        layered_images, center_points = text_guider.get_contour_center_point(self.r['masks'], 0.01)

        for index, center_point in enumerate(center_points):
            if center_point:
                # roi 를 살짝 넓직하게 잡아야 사람 포즈 인식이 잘됨
                roi = self.r['rois'][index]
                obj = Object(roi, self.r['masks'][index], self.r['class_ids'][index], self.r['scores'][index],
                             center_point)
                if obj.is_person():
                    # 사진 내 여러 사람이 있을 수 있으므로 해당 객체만을 오려내서 pose estimation 을 돌림
                    d = 30
                    roi = copy.deepcopy(roi)
                    roi[0] -= d
                    roi[1] -= d
                    roi[2] += d
                    roi[3] += d

                    height, width = self.image.shape[0], self.image.shape[1]
                    roi[0] = np.clip(roi[0], 0, width)
                    roi[1] = np.clip(roi[1], 0, height)
                    roi[2] = np.clip(roi[2], 0, width)
                    roi[3] = np.clip(roi[3], 0, height)

                    # 사람 주변으로 잘린 crop 이미지 준비
                    cropped_image = self.image[roi[0]: roi[2], roi[1]:roi[3]]

                    # 포즈 추정
                    pose = cv_estimator.inference(cropped_image)
                    if pose is not None:
                        pose_class = pose_classifier.run(pose)
                        obj = Human(obj, pose, pose_class, cropped_image, roi)

                obj_component = ObjectComponent(len(self.component_list), obj)
                # 컴포넌트 리스트에 객체 추가
                self.component_list.append(obj_component)

    def get_line_position_guides(self, line_component):
        line = line_component.line
        upper_threshold = 25
        lower_threshold = 5

        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        ydiff = abs(y1 - y2)
        xdiff = abs(x1 - x2)

        # 수평선 양 끝 점의 차이가 lower_threshold 초과이면 가이드 멘트
        if upper_threshold > ydiff > lower_threshold:
            self.guide_list[0].append(LineGuide(line_component.id, 0))

        # 수직선
        elif upper_threshold > xdiff > lower_threshold:
            self.guide_list[1].append(LineGuide(line_component.id, 1))

    def get_effective_line_and_guide(self):
        effective_lines = hough.find_hough_line(self.image)
        if effective_lines:
            for line in effective_lines:
                line_component = LineComponent(len(self.component_list), line)
                self.get_line_position_guides(line_component)

    def get_golden_ratio_area(self):
        unit_x = self.image.shape[1] // 8
        unit_y = self.image.shape[0] // 8

        area_list = list()

        # 좌상단
        area_list.append((unit_y * 2, unit_x * 2, unit_y * 3, unit_x * 3))

        # 우상단
        area_list.append((unit_y * 2, unit_x * 5, unit_y * 3, unit_x * 6))

        # 좌하단
        area_list.append((unit_y * 5, unit_x * 2, unit_y * 6, unit_x * 3))

        # 우하단
        area_list.append((unit_y * 5, unit_x * 5, unit_y * 6, unit_x * 6))

        # 정중앙
        area_list.append((unit_y * 3, unit_x * 3, unit_y * 5, unit_x * 5))
        return area_list

    def get_obj_position_guides(self, obj_component):
        image_h, image_w = self.image.shape[:2]
        error = image_w // 100

        obj = obj_component.object
        if isinstance(obj, Human):
            pose_guider = PoseGuider(obj_component)
            pose_guide_list = pose_guider.run(self.image)
            if pose_guide_list is not None:
                for pose_guide in pose_guide_list:
                    self.guide_list[pose_guide.guide_id].append(pose_guide)

        left_side = int(image_w / 3)
        right_side = int(image_w / 3 * 2)
        middle_side = int(image_w / 2)

        left_diff = int(np.abs(left_side - obj.center_point[0]))
        right_diff = int(np.abs(right_side - obj.center_point[0]))
        middle_diff = int(np.abs(middle_side - obj.center_point[0]))

        golden_ratio_area_list = self.get_golden_ratio_area()

        # for golden_ratio_area in golden_ratio_area_list:
        #     cv2.rectangle(image, (golden_ratio_area[0], golden_ratio_area[1]), (golden_ratio_area[2], golden_ratio_area[3]), (255, 0, 0))

        if left_diff < right_diff:
            if left_diff < middle_diff:
                # 왼쪽에 치우친 경우
                if left_diff > error:
                    self.guide_list[5].append(ObjectGuide(obj_component.id, 5, 0, left_side - obj.center_point[0], obj.clazz))
            else:
                # 중앙에 있는 경우
                if middle_diff > error:
                    self.guide_list[4].append(ObjectGuide(obj_component.id, 4, 0, middle_side - obj.center_point[0], obj.clazz))
        else:
            if right_diff < middle_diff:
                # 오른쪽에 치우친 경우
                if right_diff > error:
                    self.guide_list[5].append(ObjectGuide(obj_component.id, 5, 0, right_side - obj.center_point[0], obj.clazz))
            else:
                # 중앙에 있는 경우
                if middle_diff > error:
                    self.guide_list[4].append(ObjectGuide(obj_component.id, 4, 0, middle_side - obj.center_point[0], obj.clazz))


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

        # for key in Human.BODY_PARTS.keys():
        #     if human.pose[Human.BODY_PARTS[key]][2] > HumanPose.POSE_THRESHOLD:
        #         center = np.array(human.pose[Human.BODY_PARTS[key]][:2]) + np.array([human.extended_roi[1], human.extended_roi[0]])
        #         center = tuple(center)
        #         cv2.circle(image, center, 3, (255, 0, 0), thickness=3)
        #         cv2.putText(image, key, center, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 2)
        # plt.imshow(image)
        # plt.show()

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
                    human_height = self.human[2] - self.human[0]
                    diff = -human_height * 10 / 170
                    guide_message_list.append(ObjectGuide(self.human_component.id, 3, diff, 0, self.human_component.object.clazz))

                # 무릎이 잘린 경우
                if self.human.pose[Human.BODY_PARTS[Human.Part.LKnee]][2] <= HumanPose.POSE_THRESHOLD or\
                        self.human.pose[Human.BODY_PARTS[Human.Part.RKnee]][2] <= HumanPose.POSE_THRESHOLD:
                    human_height = self.human[2] - self.human[0]
                    diff = human_height * 20 / 170
                    guide_message_list.append(
                        ObjectGuide(self.human_component.id, 7, diff, 0, self.human_component.object.clazz))

                if self.has_head():
                    if not self.has_body():
                        human_height = self.human[2] - self.human[0]
                        diff = -human_height * 20 / 170
                        guide_message_list.append(
                            ObjectGuide(self.human_component.id, 8, diff, 0, self.human_component.object.clazz))

            if height > self.human.roi[2] + self.foot_lower_threshold:
                guide_message_list.append(
                    ObjectGuide(self.human_component.id, 2, height - self.human.roi[2] + self.foot_lower_threshold, 0, self.human_component.object.clazz))

            # 사람이 사진의 윗쪽에 위치한 경우
            if self.human.roi[0] < gamma:
                # 머리가 있다면
                if self.has_head():
                    top = height // 3
                    diff = top - self.human.roi[0]
                    guide_message_list.append(
                        ObjectGuide(self.human_component.id, 9, diff, 0, self.human_component.object.clazz))

        return guide_message_list