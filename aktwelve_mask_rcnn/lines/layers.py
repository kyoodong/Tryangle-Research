import numpy as np

dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]


# 주어진 segmentation 결과를 통해 외곽선을 알아내는 함수
# image : 탐색할 이미지
# x : 탐색을 시작할 x좌표
# y : 탐색을 시작할 y좌표
# visits : 해당 픽셀을 방문했는지 여부를 저장하는 2차원 boolean 배열
# threshold : threshold(0 ~ 1 사이의 값) 크기 이상의 객체만을 처리하며, 그 이하인 경우 없는 취급함
def bfs(image, x, y, visits, threshold):
    layered_image = np.zeros_like(image)
    queue = []
    queue.append((x, y))
    visits[x][y] = True

    count = 1
    while len(queue) > 0:
        position = queue.pop()

        for i in range(4):
            X = position[0] + dx[i]
            Y = position[1] + dy[i]

            if X < 0 or X >= image.shape[0] or Y < 0 or Y >= image.shape[1]:
                continue

            if image[X][Y]:
                if not visits[X][Y]:
                    visits[X][Y] = True
                    count += 1
                    queue.append((X, Y))

            else:
                layered_image[X][Y] = 1
    total = image.shape[0] * image.shape[1]
    if count > total * threshold:
        return layered_image
    return np.zeros_like(layered_image)


# 외곽선만 따는 함수
def to_lines(image, threshold):
    channels = image.shape[-1]
    width = image.shape[0]
    height = image.shape[1]
    layered_image = np.zeros_like(image)
    for i in range(channels):
        visits = np.zeros_like(image[:, :, i])
        for w in range(width):
            for h in range(height):
                if image[w][h][i] and not visits[w][h]:
                    # bfs 를 돌려서 외곽선 탐색
                    layered_image[:, :, i] += bfs(image[:, :, i], w, h, visits, threshold)
    return layered_image


