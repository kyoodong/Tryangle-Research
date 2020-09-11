
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from guide.feature_extractor import preprocess
from guide.image_retrieval import ImageRetrieval

# query Image feature 뽑는 과정
# 현재는 간단하게 MobileNetV2로 이미지의 feature를 뽑아내서 비교
input_shape = (224, 224, 3)
base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                         include_top=False,
                                         weights='imagenet')
base.trainable = False
model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

# 이미지 로드
img = preprocess('image3.jpg', input_shape)

# 이미지에서 feature 뽑아내기
fvec = model.predict(np.array([img]))

# 이미지 검색 클래스 생성
# fvecs.bin랑 fnames.txt는 feature_extractor.py에서 만들 수 있는 파일
imageRetrieval = ImageRetrieval(fvec_file="fvecs.bin",
                                fvec_img_file_name="fnames.txt")

results = imageRetrieval.search(fvec)

for path, dst in results:
    print(f"Image path: {path}, dst: {dst}")