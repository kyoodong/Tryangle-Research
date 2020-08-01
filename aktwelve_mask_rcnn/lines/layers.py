import numpy as np

dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]


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
                    layered_image[:, :, i] += bfs(image[:, :, i], w, h, visits, threshold)
    return layered_image


