

import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('images/18443993_1118340654938583_6168835005821747200_n.jpg')
dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 5000, 1500, apertureSize=5, L2gradient=True)
# plt.imshow(canny)
# plt.show()

lines = cv2.HoughLines(canny, 0.8, (np.pi) / 180, 100, srn=100, stn=200, min_theta=0, max_theta=np.pi)

for i in lines:
    rho, theta = i[0]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho

    scale = src.shape[0] + src.shape[1]

    x1 = int(x0 + scale * -b)
    y1 = int(y0 + scale * a)
    x2 = int(x0 - scale * -b)
    y2 = int(y0 - scale * a)

    cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)

plt.imshow(dst)
plt.show()
