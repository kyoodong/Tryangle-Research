from process import text_guider
from process.object import Object, Human
from process.pose import CVPoseEstimator, CvClassifier
from process.image_clusterer import find_nearest, scaling
from process.component import ObjectComponent
from process.guide import ObjectGuide
from process.guider.pose_guider import PoseGuider
import copy
import process.image_api as api
import numpy as np
import time
import matplotlib.pyplot as plt

DEBUG = True

cv_estimator = CVPoseEstimator()
pose_classifier = CvClassifier()


def get_time():
    return int(round(time.time() * 1000))


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

    def __init__(self, image, only_segmentation = True):
        self.image = image
        self.component_list = list()
        self.cluster = -1

        if DEBUG:
            plt.imshow(image)
            plt.show()

        now = get_time()
        self.r = api.segment(image)
        diff_time = get_time() - now
        print('segmentation time : ', diff_time)

        if not only_segmentation and self.r["rois"].shape[0] > 0:
            now = get_time()
            self.get_object_component()
            diff_time = get_time() - now
            print('get_object_component time : ', diff_time)

            # now = get_time()
            # self.get_effective_line_and_guide()
            # diff_time = get_time() - now
            # print('get_effective_line_and_guide time : ', diff_time)

            now = get_time()
            for component in self.component_list:
                if isinstance(component, ObjectComponent):
                    self.get_obj_position_guides(component)

            diff_time = get_time() - now
            print('get_obj_position_guides time : ', diff_time)

            if self.is_single_person():
                person = self.get_single_person()
                if isinstance(person.object, Human):
                    area = float(person.object.area) / (image.shape[0] * image.shape[1])
                    point = scaling([[
                        person.object.center_point[0],
                        person.object.center_point[1],
                        area,
                        person.object.pose[0][0], person.object.pose[0][1],
                        person.object.pose[1][0], person.object.pose[1][1],
                        person.object.pose[2][0], person.object.pose[2][1],
                        person.object.pose[3][0], person.object.pose[3][1],
                        person.object.pose[4][0], person.object.pose[4][1],
                        person.object.pose[5][0], person.object.pose[5][1],
                        person.object.pose[6][0], person.object.pose[6][1],
                        person.object.pose[7][0], person.object.pose[7][1],
                        person.object.pose[8][0], person.object.pose[8][1],
                        person.object.pose[9][0], person.object.pose[9][1],
                        person.object.pose[10][0], person.object.pose[10][1],
                        person.object.pose[11][0], person.object.pose[11][1],
                        person.object.pose[12][0], person.object.pose[12][1],
                        person.object.pose[13][0], person.object.pose[13][1],
                        person.object.pose[14][0], person.object.pose[14][1],
                        person.object.pose[15][0], person.object.pose[15][1],
                        person.object.pose[16][0], person.object.pose[16][1],
                    ]])[0]
                    self.cluster = find_nearest(point)


    def is_single_person(self):
        count = 0
        for component in self.component_list:
            if isinstance(component, ObjectComponent):
                if component.object.clazz == 0:
                    count += 1

        return count == 1

    def get_single_person(self):
        for component in self.component_list:
            if isinstance(component, ObjectComponent):
                if component.object.clazz == 0:
                    return component

    def get_object_component(self):
        # 객체마다 외곽선만 따도록 수정
        # [image_height][image_width][num_of_obj]
        # 위와 같은 shape 로 이미지가 처리되며 num_of_obj 개수로 나뉜 이미지들을 하나의 이미지로 합쳐야함
        now = get_time()
        layered_images, center_points, areas = text_guider.get_contour_center_point(self.r['masks'], 0.01)
        diff = get_time() - now
        print("\tget_contour_center_point time ", diff)

        for index, center_point in enumerate(center_points):
            if center_point:
                if DEBUG:
                    plt.imshow(layered_images[index], 'gray')
                    plt.show()

                # roi 를 살짝 넓직하게 잡아야 사람 포즈 인식이 잘됨
                roi = self.r['rois'][index]
                obj = Object(roi, self.r['masks'][index, :, :], self.r['class_ids'][index], self.r['scores'][index],
                             center_point, areas[index])
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

                    if DEBUG:
                        plt.imshow(cropped_image)
                        plt.show()

                    # 포즈 추정
                    now = get_time()
                    pose = cv_estimator.inference(cropped_image)
                    diff = get_time() - now
                    print("\tpose_estimation time ", diff)
                    if pose is not None:
                        now = get_time()
                        pose_class = pose_classifier.run(pose)
                        diff = get_time() - now
                        print("\tpose classify time ", diff)
                        obj = Human(obj, pose, pose_class, cropped_image, roi)

                obj_component = ObjectComponent(len(self.component_list), obj)
                # 컴포넌트 리스트에 객체 추가
                self.component_list.append(obj_component)

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
                    obj_component.guide_list.append(pose_guide)

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
                    self.guide_list[5].append(ObjectGuide(obj_component.id, 5, 0, left_side - obj.center_point[0]))
            else:
                # 중앙에 있는 경우
                if middle_diff > error:
                    self.guide_list[4].append(ObjectGuide(obj_component.id, 4, 0, middle_side - obj.center_point[0]))
        else:
            if right_diff < middle_diff:
                # 오른쪽에 치우친 경우
                if right_diff > error:
                    self.guide_list[5].append(ObjectGuide(obj_component.id, 5, 0, right_side - obj.center_point[0]))
            else:
                # 중앙에 있는 경우
                if middle_diff > error:
                    self.guide_list[4].append(ObjectGuide(obj_component.id, 4, 0, middle_side - obj.center_point[0]))
