
import cv2
import numpy as np
from skimage import feature, transform

#########################################
# 소실점 찾기
#########################################

def find_vps(image):
    if isinstance(image, str):
        image = cv2.imread(image, -1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lines, edgelets = __detect_lines(image)
    vanishing_point = __compute_vp(lines, edgelets, ransac_iter=2000)

    return vanishing_point


def __detect_lines(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 이미지 밝기 조절
    clahe = cv2.createCLAHE(2.0, (8, 8))
    clahe_image = clahe.apply(gray_img)

    # 이미지에서 선을 찾는 부분
    canny_img = feature.canny(clahe_image, 3).astype(np.uint8)
    lines = transform.probabilistic_hough_line(canny_img, line_length=10, line_gap=3)
    lines = np.reshape(lines, newshape=(-1, 4))

    # calc edgelet
    locations = []
    directions = []
    strengths = []
    for x1, y1, x2, y2 in lines:
        p0, p1 = np.array([x1, y1]), np.array([x2, y2])
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / \
                 np.linalg.norm(directions, axis=1)[:, np.newaxis]

    edgelets = [locations, directions, strengths]

    return lines, edgelets


def __compute_vp(lines, edgelets, ransac_iter):
    # 선분의 두 점 외적
    N = lines.shape[0]

    p1 = np.column_stack((lines[:, :2], np.ones(N,
                                                dtype=np.float32)))
    p2 = np.column_stack((lines[:, 2:], np.ones(N,
                                                dtype=np.float32)))

    cross_p = np.cross(p1, p2)
    cross_p_len = len(cross_p)

    best_vp = None
    best_votes = np.zeros(cross_p_len)

    for i in range(ransac_iter):
        idx1 = np.random.randint(cross_p_len)
        idx2 = np.random.randint(cross_p_len)

        vp_candidate = np.cross(cross_p[idx1], cross_p[idx2])

        if np.sum(vp_candidate ** 2) < 1 or vp_candidate[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = __compute_vote(edgelets, vp_candidate)

        if current_votes.sum() > best_votes.sum():
            best_vp = vp_candidate
            best_votes = current_votes

    return best_vp


def __compute_vote(edgelets, vp_candidate, threshold_inlier=5):
    vp = vp_candidate[:2] / vp_candidate[2]

    locations, directions, strengths = edgelets

    # 소실점에서 선분의 중점으로 향하는 벡터
    est_directions = vp - locations

    # 내적 연산
    dot_prod = np.sum(est_directions * directions, axis=1)

    # direction의 길이 * 위의 est_direction의 길이
    abs_prod = np.linalg.norm(directions, axis=1) * \
               np.linalg.norm(est_directions, axis=1)

    # 두 길이의 곱이 0이면 1e-5로 결정
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    cosine_theta = np.clip(cosine_theta, -1., 1.)
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths ** 2
