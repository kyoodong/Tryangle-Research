import numpy as np
import cv2
import glob

from retrieval.utils import drawer
from retrieval.vp_detection import find_vps


test_image = [4]
imgs = []
for i in test_image:
    path = f"image/test/image{i}.jpg"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

vp = find_vps(imgs[0])

drawer.display("Query",
               imgs[0],
               vanishing_point=vp)
