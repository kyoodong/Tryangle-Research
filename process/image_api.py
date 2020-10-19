import cv2
import numpy as np

from process.segmentation import YOLACT

model = YOLACT()


def segment(image):
    """

    :param image: image numpy array from cv2.imread
    """

    ### 밝기 보정 소스
    gamma = 0.4
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)

    ### CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 현재까지는 clahe 가 가장 보기 좋음
    # 적어도 effective line 을 찾기에는 유용함
    image = clahe_image

    # Run detection
    results = model.detect(image)
    if results["rois"].shape[0] > 0:
        tmp = results["rois"][:, 1].copy()
        results["rois"][:, 1] = results["rois"][:, 0].copy()
        results["rois"][:, 0] = tmp.copy()

        tmp = results["rois"][:, 3].copy()
        results["rois"][:, 3] = results["rois"][:, 2].copy()
        results["rois"][:, 2] = tmp.copy()
    return results