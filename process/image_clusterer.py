import cv2
import pandas as pd
import random
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

df = pd.read_csv('{}/process/cluster_data.csv'.format(ROOT_DIR), names=['url', 'image_id', 'center_x', 'center_y', 'area', 'id', 'object_id', 'pose_id',
                                    'nose_x', 'nose_y',
                                    'left_eye_x', 'left_eye_y',
                                    'right_eye_x', 'right_eye_y',
                                    'left_ear_x', 'left_ear_y',
                                    'right_ear_x', 'right_ear_y',
                                    'left_shoulder_x', 'left_shoulder_y',
                                    'right_shoulder_x', 'right_shoulder_y',
                                    'left_elbow_x', 'left_elbow_y',
                                    'right_elbow_x', 'right_elbow_y',
                                    'left_wrist_x', 'left_wrist_y',
                                                       'right_wrist_x', 'right_wrist_y',
                                                       'left_heap_x', 'left_heap_y',
                                                       'right_heap_x', 'right_heap_y',
                                                       'left_knee_x', 'left_knee_y',
                                                       'right_knee_x', 'right_knee_y',
                                                       'left_ankle_x', 'left_ankle_y',
                                                       'right_ankle_x', 'right_ankle_y'])
df.head()
print('데이터 파일 읽기 성공')

c = df.drop(['url', 'id', 'image_id', 'pose_id', 'object_id'], axis=1)
y = df['url'].values

scaler = StandardScaler()
x = scaler.fit_transform(c)

features = ['center_x', 'center_y', 'area', 'nose_x', 'nose_y',
                                    'left_eye_x', 'left_eye_y',
                                    'right_eye_x', 'right_eye_y',
                                    'left_ear_x', 'left_ear_x',
                                    'right_ear_x', 'right_ear_x',
                                    'left_shoulder_x', 'left_shoulder_x',
                                    'right_shoulder_x', 'right_shoulder_x',
                                    'left_elbow_x', 'left_elbow_x',
                                    'right_elbow_x', 'right_elbow_x',
                                    'left_wrist_x', 'left_wrist_x',
                                    'right_wrist_x', 'right_wrist_x',
                                    'left_heap_x', 'left_heap_x',
                                    'right_heap_x', 'right_heap_x',
                                    'left_knee_x', 'left_knee_x',
                                    'right_knee_x', 'right_knee_x',
                                    'left_ankle_x', 'left_ankle_x',
                                    'right_ankle_x', 'right_ankle_x']
pd.DataFrame(x, columns=features).head()

columns = ['component{}'.format(i) for i in range(1, len(features) + 1)]
principalDF = pd.DataFrame(data=x, columns=columns)

Z = principalDF.values
N = 15

print("KMeans 실행 {}".format(N))
kmeans = KMeans(n_clusters=N).fit(Z)


def scaling(point):
    return scaler.transform(point)


def find_nearest(point):
    min_index = -1
    min_value = 987654321
    target = point

    for i in range(kmeans.cluster_centers_.shape[0]):
        center = kmeans.cluster_centers_[i]
        value = 0
        for j in range(center.shape[0]):
            value += (target[j] - center[j]) ** 2
        value = math.sqrt(value)

        if min_value > value:
            min_index = i
            min_value = value

    return min_index


