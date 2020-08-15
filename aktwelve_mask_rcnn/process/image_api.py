import numpy as np
import cv2
import os
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common

dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]

estimator = TfPoseEstimator(get_graph_path('cmu'), target_size=(432, 368))
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


def recommend_object_position(center_point, image, is_person=False):
    if is_person:
        humans = estimator.inference(image, upsample_size=4.0)
        image_h, image_w = image.shape[:2]
        centers = {}
        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                # cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
        print(humans)
    return ''

