import numpy as np
import cv2
import sys
import os
from sys import platform
import argparse

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
    x_count = 1

    while len(queue) > 0:
        position = queue.pop()

        for i in range(4):
            X = position[1] + dx[i]
            Y = position[0] + dy[i]

            if X < 0 or X >= image.shape[0] or Y < 0 or Y >= image.shape[1]:
                continue

            if image[Y][X]:
                if not visits[Y][X]:
                    visits[Y][X] = True
                    count += 1
                    queue.append((Y, X))

            else:
                # 무게 중심을 구하기 위한 contour 수집
                contour.append((X, Y))

                # 질량 중심을 구하기 위한 각 좌표 합
                x_sum += X
                y_sum += Y
                x_count += 1

                layered_image[Y][X] = 1
    total = image.shape[0] * image.shape[1]
    if count > total * threshold:
        ### 무게 중심
        # M = cv2.moments(np.array(contour))
        # cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])

        ### 질량 중심
        cx = int(x_sum / x_count)
        cy = int(y_sum / x_count)
        return layered_image, (cx, cy)
    return np.zeros_like(layered_image), None


# 외곽선 + 오브젝트의 무게중심을 구해주는 함수
def get_contour_center_point(image, threshold):
    channels = image.shape[-1]
    width = image.shape[0]
    height = image.shape[1]
    layered_image = np.zeros_like(image)
    cogs = list()
    for i in range(channels):
        visits = np.zeros_like(image[:, :, i])
        for w in range(width):
            for h in range(height):
                if image[h][w][i] and not visits[h][w]:
                    # bfs 를 돌려서 외곽선 탐색
                    l, cog = __get_contour_center_point(image[:, :, i], w, h, visits, threshold)
                    layered_image[:, :, i] += l
                    cogs.append(cog)
    return layered_image, cogs


def recommend_object_position(center_point, image, is_person=False):
    print('center_point', center_point)
    print('image', image.shape)
    print('is_person', is_person)

    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                # 윈도우 코드는 테스트 안해봐서 안될 수도 있음 경로를 적절히 수정해줘야함...
                sys.path.append(dir_path + '../openpose/python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../openpose/build/python')

                print(os.getcwd())
                for path in sys.path:
                    print(path)
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg",
                            help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../openpose/models/"

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1]) - 1:
                next_item = args[1][i + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        # imageToProcess = cv2.imread(image)
        imageToProcess = image
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(0)
    except Exception as e:
        print(e)
        sys.exit(-1)
    return ""
