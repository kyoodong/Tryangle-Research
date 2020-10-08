import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


e = TfPoseEstimator(get_graph_path("cmu"), target_size=(432, 368))
image = common.read_imgfile("../images/test11.jpg", None, None)
humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()