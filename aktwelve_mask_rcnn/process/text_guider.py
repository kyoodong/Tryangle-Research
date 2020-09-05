import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from process.pose import CVPoseEstimator, PoseGuider, CvClassifier, Pose
from process.object import Object, Human
import numpy as np

dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]


# 주어진 segmentation 결과를 통해 외곽선을 알아내는 함수
# image : 탐색할 이미지
# x : 탐색을 시작할 x좌표
# y : 탐색을 시작할 y좌표
# visits : 해당 픽셀을 방문했는지 여부를 저장하는 2차원 boolean 배열
# threshold : threshold(0 ~ 1 사이의 값) 크기 이상의 객체만을 처리하며, 그 이하인 경우 없는 취급함
def __get_contour_center_point(image, x, y, visits, threshold):
    layered_image = np.zeros_like(image)
    queue = []
    queue.append((y, x))
    visits[y][x] = True

    count = 1

    # 무게 중심 변수
    contour = list()

    # 질량 중심 변수
    x_sum = x
    y_sum = y

    while len(queue) > 0:
        position = queue.pop()

        for i in range(4):
            Y = position[0] + dy[i]
            X = position[1] + dx[i]

            if X < 0 or X >= image.shape[1] or Y < 0 or Y >= image.shape[0]:
                continue

            if image[Y][X]:
                if not visits[Y][X]:
                    visits[Y][X] = True
                    count += 1
                    queue.append((Y, X))

                    # 질량 중심을 구하기 위한 각 좌표 합
                    x_sum += X
                    y_sum += Y

            else:
                # 무게 중심을 구하기 위한 contour 수집
                contour.append((Y, X))
                layered_image[Y][X] = 1
    total = image.shape[0] * image.shape[1]
    if count > total * threshold:
        ### 무게 중심
        # M = cv2.moments(np.array(contour))
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])

        ### 질량 중심
        cx = int(x_sum / count)
        cy = int(y_sum / count)
        return layered_image, (cx, cy)
    return np.zeros_like(layered_image), None


# 외곽선 + 오브젝트의 무게중심을 구해주는 함수
def get_contour_center_point(image, threshold):
    channels = image.shape[-1]
    height = image.shape[0]
    width = image.shape[1]
    layered_image = np.zeros_like(image)
    cogs = list()
    for i in range(channels):
        visits = np.zeros_like(image[:, :, i])
        is_finish = False
        for h in range(height):
            for w in range(width):
                if image[h][w][i] and not visits[h][w]:
                    # bfs 를 돌려서 외곽선 탐색
                    l, cog = __get_contour_center_point(image[:, :, i], w, h, visits, threshold)
                    layered_image[:, :, i] += l
                    cogs.append(cog)
                    is_finish = True
                    break
            if is_finish:
                break
    return layered_image, cogs


def get_golden_ratio_area(image_height, image_width):
    unit_x = image_width // 8
    unit_y = image_height // 8

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


def get_iou(rect1, rect2):
    rect1_width = rect1[3] - rect1[1]
    rect1_height = rect1[2] - rect1[0]
    rect1_area = rect1_width * rect1_height
    intersection_left = max(rect1[1], rect2[1])
    intersection_right = min(rect1[3], rect2[3])
    intersection_top = max(rect1[0], rect2[0])
    intersection_bottom = min(rect1[2], rect2[2])
    intersection_width = intersection_right - intersection_left
    intersection_height = intersection_bottom - intersection_top
    intersection_area = max(intersection_width * intersection_height, 0)
    if intersection_width < 0 or intersection_height < 0:
        intersection_area = 0
    return intersection_area / rect1_area


def get_obj_position_guides(obj, image):
    image_h, image_w = image.shape[:2]
    error = image_w // 100
    guide_message_list = list()

    if obj.is_person():
        pose_guider = PoseGuider()
        pose_guide = pose_guider.run(obj, image)
        if pose_guide:
            guide_message_list.extend(pose_guide)

    left_side = int(image_w / 3)
    right_side = int(image_w / 3 * 2)
    middle_side = int(image_w / 2)

    left_diff = int(np.abs(left_side - obj.center_point[0]))
    right_diff = int(np.abs(right_side - obj.center_point[0]))
    middle_diff = int(np.abs(middle_side - obj.center_point[0]))

    golden_ratio_area_list = get_golden_ratio_area(image_h, image_w)

    # for golden_ratio_area in golden_ratio_area_list:
    #     cv2.rectangle(image, (golden_ratio_area[0], golden_ratio_area[1]), (golden_ratio_area[2], golden_ratio_area[3]), (255, 0, 0))

    if left_diff < right_diff:
        if left_diff < middle_diff:
            # 왼쪽에 치우친 경우
            if left_diff > error:
                guide_message_list.append(("피사체가 황금비율 영역에 있어야 좋습니다.", (0, left_side - obj.center_point[0])))
        else:
            # 중앙에 있는 경우
            if middle_diff > error:
                guide_message_list.append(("피사체를 정중앙에 두어 좌우 대칭을 맞춰보세요", (0, middle_side - obj.center_point[0])))
    else:
        if right_diff < middle_diff:
            # 오른쪽에 치우친 경우
            if right_diff > error:
                guide_message_list.append(("피사체가 황금비율 영역에 있어야 좋습니다.", (0, right_side - obj.center_point[0])))
        else:
            # 중앙에 있는 경우
            if middle_diff > error:
                guide_message_list.append(("피사체를 정중앙에 두어 좌우 대칭을 맞춰보세요", (0, middle_side - obj.center_point[0])))

    return guide_message_list


def shift_image(image, x, y):
    height, width = image.shape[:2]
    M = np.float32([[1, 0, x], [0, 1, y]])
    dst = cv2.warpAffine(np.float32(image), M, (width, height))
    return dst


def get_line_position_guides(line):
    upper_threshold = 25
    lower_threshold = 5
    guide_message_list = list()

    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    ydiff = abs(y1 - y2)
    xdiff = abs(x1 - x2)

    # 수평선 양 끝 점의 차이가 lower_threshold 초과이면 가이드 멘트
    if upper_threshold > ydiff > lower_threshold:
        guide_message_list.append("수평을 맞춰주세요")

    # 수직선
    elif upper_threshold > xdiff > lower_threshold:
        guide_message_list.append("수직을 맞춰주세요")

    return guide_message_list


def get_obj_line_guides(objs, lines, image):
    guide_message_list = list()
    threshold = 10
    for obj in objs:
        if obj.is_person():
            # joint_list = [("Neck", "목", "어깨")]
            joint_list = []

            # 선이 관절을 지나는지 검사
            for joint in joint_list:
                if obj.pose[Human.BODY_PARTS[joint[0]]]:
                    for line in lines:
                        neck = np.array([obj.pose[Human.BODY_PARTS[joint[0]]][0] + obj.roi[1],
                                         obj.pose[Human.BODY_PARTS[joint[0]]][1] + obj.roi[0]])

                        line_p1 = np.array([line[0], line[1]])
                        line_p2 = np.array([line[2], line[3]])
                        distance = np.linalg.norm(np.cross(line_p2 - line_p1, line_p1 - neck)) / np.linalg.norm(
                            line_p2 - line_p1)
                        if distance < threshold:
                            guide_message_list.append("선이 {}을 지나는 것은 좋지 않습니다. {}를 지나게 해보세요".format(joint[1], joint[2]))
    return guide_message_list
