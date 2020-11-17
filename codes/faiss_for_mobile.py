import pymysql
import os
import sys
import shutil
from retrieval.feature.pt_extractor import extract

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

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

sql = '''
SELECT url FROM tryangle_image
WHERE score >= 4
'''
cursor.execute(sql)
url_list = [r[0] for r in cursor.fetchall()]
print(url_list)

base_path = '/home/dongkyoo/Develop/gomson-3/TryangleAppServer/build/resources/main/images'
target_base_dir_path = './mobile_feature'
target_dir_path = '{}/images'.format(target_base_dir_path)
feature_target_dir_path = '{}/features'.format(target_base_dir_path)

if not os.path.exists(target_base_dir_path):
    os.mkdir(target_base_dir_path)
    os.mkdir(target_dir_path)
    os.mkdir(feature_target_dir_path)

    for index, url in enumerate(url_list):
        image_path = '{}/{}'.format(base_path, url)
        target_path = '{}/{}'.format(target_dir_path, url)
        shutil.copyfile(image_path, target_path)

        if index % 100 == 0:
            print('{} / {} completed'.format(index, len(url_list)))

extract(image_dataset=target_base_dir_path,
            store_dir=feature_target_dir_path,
            store_file='fvecs')

