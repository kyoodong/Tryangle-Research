

import cv2
import numpy as np
import guide.utils
from guide import ciede2000 as color_diff

from skimage import feature, transform
from matplotlib import pyplot as plt
from matplotlib import patches as patches


def find_similar_by_CBIR(gi, other_gi, count=3):

    return None

def find_similar_by_SIFT(gi, other_gi, count=3):

    return None

def find_similar_by_color(gi, other_gi, count=3):
    if count > len(other_gi):
        count = len(other_gi)

    diff = diff_color(gi, other_gi)
    diff_arg = np.argsort(diff)

    return diff_arg[:count]

def diff_color(gi, other_gi):
    # 방법 1
    # 각 이미지의 주요 색끼리 ciede2000를 이용하여 차이를 구해고

    diff = []
    ciede2000 = color_diff.ciede2000
    rgb2lab = color_diff.rgb2lab
    for ogi in other_gi:
        ogi_diff_sum = 0
        for ocp in ogi.dominant_color_pallets:
            ogi_diff_sum += np.min([ciede2000(rgb2lab(cp), rgb2lab(ocp)) for cp in gi.dominant_color_pallets])
        diff.append(ogi_diff_sum)
    return np.array(diff)


##########################################
# Guide Image
##########################################
class GuideImage:
    class ColorCategory:
        NAMES = [
            "Yellow",
            "Yellow_orange",
            "Orange",
            "Red_orange",
            "Red",
            "Pink",
            "Red_violet",
            "Violet",
            "Blue_violet",
            "Blue",
            "Sky",
            "Blue_green",
            "Green",
            "Yellow_green",
            "White",
            "Gray",
            "Brown",
            "Black",
        ]

        LIST = [(246, 239, 30),  # Yellow
                (249, 197, 14),  # Yellow_orange
                (244, 125, 25),  # Orange
                (233, 62, 29),  # Red_orange
                (229, 0, 29),  # Red
                (253, 181, 211),  # Pink
                (139, 41, 134),  # Red_violet
                (100, 27, 128),  # Violet
                (81, 69, 152),  # Blue_violet
                (49, 80, 162),  # Blue
                (135, 198, 228),  # Sky
                (28, 128, 107),  # Blue_green
                (45, 170, 64),  # Green
                (127, 191, 51),  # Yellow_green
                (255, 255, 255),  # White
                (186, 186, 186),  # Gray
                (125, 69, 36),  # Brown
                (0, 0, 0),  # Black
                ]

    def __init__(self, img):
        if isinstance(img, str):
            img = cv2.imread(img, -1)

        self.__img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Keep a copy for later

        # 색상 정보
        self.dominant_color_pallets = None

        # 소실점 정보
        self.vanishing_point = None
        self.__edgelets = None
        self.__lines = None

        # 오브젝트 정보

    def compute_info(self, n_color=3, image_processing_size=(32,32)):
        self.__find_vps()
        self.__set_dominant_color(n_color=n_color, image_processing_size=image_processing_size)

    def visualize(self,
                  vp_vis=False,
                  color_vis=False,
                  lines_vis=False):
        # 주요 색
        if color_vis:
            fig, ax = plt.subplots()
            width = 1 / len(self.dominant_color_pallets)
            for i, color_p in enumerate(self.dominant_color_pallets):
                ax.add_patch(
                    patches.Rectangle(
                        (width * i, 0),
                        width, 1,
                        facecolor=color_p / 255,
                        fill=True
                    ))

            plt.xticks([]), plt.yticks([])
            plt.show()

        # 이미지
        plt.imshow(self.__img)
        plt.xticks([]), plt.yticks([])
        if vp_vis:
            self.vanishing_point[:2] /= self.vanishing_point[2]
            plt.plot(int(self.vanishing_point[0]), int(self.vanishing_point[1]), 'go')
        plt.show()

        # 이미지의 선을 보여줌
        if lines_vis:
            only_line = utils.draw_line(np.zeros_like(self.__img), self.__lines, color=(255, 0, 0), thickness=2)
            plt.imshow(only_line, cmap='gray')
            plt.xticks([]), plt.yticks([])

            plt.show()


    #########################################
    # 주요 색 찾기
    #########################################
    def __set_dominant_color(self, n_color, image_processing_size):
        image = cv2.resize(self.__img, image_processing_size)
        pixels = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        a, labels, pallets = cv2.kmeans(pixels, n_color, None, criteria, 10, flags)
        b, counts = np.unique(labels, return_counts=True)
        counts_argsort = np.argsort(counts)[::-1]

        self.dominant_color_pallets = pallets[counts_argsort]


    #########################################
    # 소실점 찾기
    #########################################

    def __find_vps(self):

        print("__detect_lines")
        edgelets = self.__detect_lines()

        self.vanishing_point = self.__compute_vp(ransac_iter=2000)

    def __detect_lines(self):

        gray_img = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)
        # 이미지 밝기 조절
        clahe = cv2.createCLAHE(2.0, (8, 8))
        clahe_image = clahe.apply(gray_img)

        # 이미지에서 선을 찾는 부분
        canny_img = feature.canny(clahe_image, 3).astype(np.uint8)
        lines = transform.probabilistic_hough_line(canny_img, line_length=10, line_gap=3)
        lines = np.reshape(lines, newshape=(-1, 4))

        self.__lines = lines

        # calc edgelet
        locations = []
        directions = []
        strengths = []
        for x1, y1, x2, y2 in self.__lines:
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

        self.__edgelets = [locations, directions, strengths]
        return self.__edgelets

    def __compute_vp(self, ransac_iter):

        # 선분의 두 점 외적
        N = self.__lines.shape[0]

        p1 = np.column_stack((self.__lines[:, :2], np.ones(N,
                                                           dtype=np.float32)))
        p2 = np.column_stack((self.__lines[:, 2:], np.ones(N,
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

            current_votes = self.__compute_vote(vp_candidate)

            if current_votes.sum() > best_votes.sum():
                best_vp = vp_candidate
                best_votes = current_votes

        return best_vp

    def __compute_vote(self, vp_candidate, threshold_inlier=5):

        vp = vp_candidate[:2] / vp_candidate[2]

        locations, directions, strengths = self.__edgelets

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
