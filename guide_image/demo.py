import numpy as np
import cv2
import glob

from guide import dominant_color
from guide.utils import drawer
from guide.vp_detection import find_vps


test_image = [4]
dominant_colors = []
for i in test_image:
    path = f"image/test/image{i}.jpg"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = dominant_color.get_dominant_color(img)
    dominant_colors.append([img, color])

vp = find_vps(dominant_colors[0][0])

drawer.display("Query",
               dominant_colors[0][0],
               vanishing_point=vp)
drawer.draw_color(dominant_colors[0][1])

# 색감이 비슷한 사진 찾기
list_img = []
for file in glob.glob("image/similar_test/*.jpg"):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = dominant_color.get_dominant_color(img)
    list_img.append([img, color])


similar_img_arg = dominant_color.find_similar_by_color(dominant_colors[0][1], np.array(list_img)[:,1], count=5)

for arg in similar_img_arg:
    drawer.display(f"Result{arg}", list_img[arg][0])
    drawer.draw_color(list_img[arg][1])

