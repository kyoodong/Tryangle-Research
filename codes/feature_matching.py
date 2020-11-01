import cv2 as cv
import os
import sys
from process.guider.guider import Guider
from process.time import get_millisecond

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
while True:
    file1 = input('파일 1 : ')
    file1 += '.jpg'
    image1 = cv.imread(IMAGE_DIR + '/' + file1)
    img1 = cv.imread(IMAGE_DIR + '/' + file1, cv.IMREAD_GRAYSCALE)  # queryImage

    guider1 = Guider(image1)
    obj = img1[guider1.r['rois'][0][0]:guider1.r['rois'][0][2], guider1.r['rois'][0][1]:guider1.r['rois'][0][3]]

    for i in range(1, 25):
        file2 = 'test{}.jpg'.format(i)
        image2 = cv.imread(IMAGE_DIR + '/' + file2)
        img2 = cv.imread(IMAGE_DIR + '/' + file2, cv.IMREAD_GRAYSCALE)  # trainImage

        total = get_millisecond()

        # Initiate SIFT detector
        t1 = get_millisecond()
        sift = cv.SIFT_create()
        t2 = get_millisecond()
        print('sift', t2 - t1)
        # find the keypoints and descriptors with SIFT
        t1 = get_millisecond()
        kp1, des1 = sift.detectAndCompute(obj, None)
        t2 = get_millisecond()
        print('detect1', t2 - t1)
        t1 = get_millisecond()
        kp2, des2 = sift.detectAndCompute(img2, None)
        t2 = get_millisecond()
        print('detect2', t2 - t1)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        t1 = get_millisecond()
        flann = cv.FlannBasedMatcher(index_params, search_params)
        t2 = get_millisecond()
        print('Flann', t2 - t1)
        t1 = get_millisecond()
        matches = flann.knnMatch(des1, des2, k=2)
        t2 = get_millisecond()
        print('knnMatch', t2 - t1)
        t1 = get_millisecond()
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        count = 0
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                count += 1
        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=(255, 0, 0),
        #                    matchesMask=matchesMask,
        #                    flags=cv.DrawMatchesFlags_DEFAULT)
        # img3 = cv.drawMatchesKnn(obj, kp1, img2, kp2, matches, None, **draw_params)
        # plt.imshow(img3)
        # plt.show()
        total = get_millisecond() - total
        print('{} : {}, {} ms'.format(file2, count, total))
