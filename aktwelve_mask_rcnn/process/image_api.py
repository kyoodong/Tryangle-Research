from enum import Enum
import numpy as np
import cv2
import os
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common
import matplotlib.pyplot as plt

dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]

estimator = TfPoseEstimator(get_graph_path('mobilenet_v2_large'))
print('estimator is ready')


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


class HumanPose(Enum):
    Unknown = 0
    Stand = 1
    Sit = 2


def recommend_object_position(center_point, image, roi, is_person=False):
    image_h, image_w = image.shape[:2]
    error = image_w // 100
    recommendation_text_list = list()

    if is_person:
        # 사진 내 여러 사람이 있을 수 있으므로 해당 객체만을 오려내서 pose estimation 을 돌림
        cropped_image = image[roi[0]:roi[2], roi[1]:roi[3]]

        humans = estimator.inference(cropped_image, upsample_size=8.0s)
        print(humans)
        human_pose = HumanPose.Unknown

        # test1.jpg 같은 경우 뒷 모습이라 그런가 pose-estimation 이 안먹음
        for human in humans:
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
            if (left_knee != -1 and left_hip != -1 and left_knee > left_hip + gamma) or\
                    (right_knee != -1 and right_hip != -1 and right_knee > right_hip + gamma):
                human_pose = HumanPose.Stand
                break

            # draw point
            # for i in range(common.CocoPart.Background.value):
            #     if i not in human.body_parts.keys():
            #         continue
            #
            #     body_part = human.body_parts[i]
            #     center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            #     centers[i] = center
            #     cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

        if human_pose == HumanPose.Stand:
            # 사람 roi의 맨 밑 선을 발 끝이라고 판단
            foot_position = roi[2]
            foot_gamma = 10
            if image_h > foot_position + foot_gamma:
                recommendation_text_list.append(("발 끝을 사진 맨 밑에 맞추세요",
                                                 (center_point[0], center_point[1] + image_h - foot_position)))


    left_side = int(image_w / 3)
    right_side = int(image_w / 3 * 2)
    middle_side = int(image_w / 2)

    left_diff = int(np.abs(left_side - center_point[0]))
    right_diff = int(np.abs(right_side - center_point[0]))
    middle_diff = int(np.abs(middle_side - center_point[0]))

    if left_diff < right_diff:
        if left_diff < middle_diff:
            # 왼쪽에 치우친 경우
            if left_diff > error:
                recommendation_text_list.append(("삼분할법을 지키세요", (left_side, center_point[1])))
        else:
            # 중앙에 있는 경우
            if middle_diff > error:
                recommendation_text_list.append(("좌우 대칭을 맞춰주세요", (middle_side, center_point[1])))
    else:
        if right_diff < middle_diff:
            # 오른쪽에 치우친 경우
            if right_diff > error:
                recommendation_text_list.append(("삼분할법을 지키세요", (right_side, center_point[1])))
        else:
            # 중앙에 있는 경우
            if middle_diff > error:
                recommendation_text_list.append(("좌우 대칭을 맞춰주세요", (middle_side, center_point[1])))

    return recommendation_text_list


def recommend_line_position(line):
    upper_threshold = 25
    lower_threshold = 5
    recommendation_message_list = list()

    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    ydiff = abs(y1 - y2)
    xdiff = abs(x1 - x2)

    # 수평선 양 끝 점의 차이가 lower_threshold 초과이면 가이드 멘트
    if upper_threshold > ydiff > lower_threshold:
        recommendation_message_list.append("수평을 맞춰주세요")

    # 수직선
    elif upper_threshold > xdiff > lower_threshold:
        recommendation_message_list.append("수직을 맞춰주세요")

    return recommendation_message_list
