import numpy as np
from guide import guide_image as gi
import guide.utils
import cv2
import glob


test_image = np.arange(1, 15)
guide_imgs = []
for i in test_image:
    guide_img = gi.GuideImage('image/test/image{}.jpg'.format(i))
    guide_img.compute_info(n_color=4)
    guide_imgs.append(guide_img)

# 이미지들 출력
for guide_img in guide_imgs:
    guide_img.visualize(vp_vis=True)

# 색감이 비슷한 사진 찾기
list_img = []
for file in glob.glob("image/similar_test/*.jpg"):
    guide_img = gi.GuideImage(file)
    guide_img.compute_info(n_color=4)
    list_img.append(guide_img)


similar_img_arg = gi.find_similar_by_color(guide_imgs[0], list_img, count=5)

for arg in similar_img_arg:
    list_img[arg].visualize(color_vis=True)