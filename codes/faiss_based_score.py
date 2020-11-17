import cv2
from retrieval.image_retrieval import ImageRetrieval
import pymysql
import os
from os import listdir
from os.path import isfile, join
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
import pandas as pd

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = "/home/dongkyoo/Develop/gomson-3/TryangleAppServer/build/resources/main/images"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


db = pymysql.connect(
    host='localhost',
    port=3306,
    user='gomson_admin',
    password='Ga123!@#',
    db='tryangle',
    charset='utf8'
)

cursor = db.cursor()


# 이미지 전처리 프로세싱
def preprocess(image, input_shape=None):
    img = tf.convert_to_tensor(image)
    # img = tf.image.decode_jpeg(img, channels=input_shape[2])
    if input_shape is not None:
        img = tf.image.resize(img, input_shape[:2])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


# index_types = ['hnsw', 'l2', 'IVFFlat', 'IVFPQ']
index_types = ['hnsw']
# image_counts = [i for i in range(10, 200, 10)]
image_counts = [5]

input_shape = (224, 224, 3)
base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                         include_top=False,
                                         weights='imagenet')
base.trainable = False
model = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))

for index_type in index_types:
    dim = 1280
    image_retrieval = ImageRetrieval(fvec_file='../retrieval/feature/output/fvecs.bin',
                                     fvec_img_file_name='../retrieval/feature/output/fvecs_names.txt',
                                     fvec_dim=dim,
                                     index_type=index_type)

    plt.title(index_type)
    data_list = []

    for image_count in image_counts:
        df = pd.read_csv('score_test.csv')

        all_count = 0
        correct_count = 0
        incorrect_count = 0

        for index, row in df.iterrows():
            url = row[0]
            score = row[1]

            filename = '{}/{}'.format(IMAGE_DIR, url)
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = preprocess(image, input_shape)
            image = tf.reshape(image, (1,) + input_shape)
            image_vec = model.predict(image)
            results = image_retrieval.search(image_vec, image_count)

            urls = [result[0].split('/')[-1] for result in results]
            url_list_sql = str(urls)

            sql = """
            SELECT score FROM tryangle_image WHERE url IN ({}) AND score >= 0;
            """.format(url_list_sql.strip('[]'))

            cursor.execute(sql)
            results = cursor.fetchall()
            scores = [int(result[0]) for result in results]
            if len(scores) == 0:
                continue

            total_score = 0

            for i, score in enumerate(scores):
                weight = len(scores) - i
                total_score += weight * score

            index_sum = sum([i for i in range(len(scores) + 1)])
            avg_score = int(float(total_score) / index_sum)
            all_count += 1
            if avg_score == score:
                correct_count += 1
            else:
                incorrect_count += 1

            if all_count % 10 == 0:
                print('{} / {} processed!'.format(all_count, len(df)))

        percent = correct_count / all_count * 100
        print('{}% success!'.format(percent))
        data_list.append(percent)

    plt.bar(range(len(data_list)), data_list)
    plt.show()
