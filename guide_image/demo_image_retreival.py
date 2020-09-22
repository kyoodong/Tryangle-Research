import numpy as np
import time

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


from guide.feature_extractor import preprocess
from guide.image_retrieval import ImageRetrieval


def retrieval_res(image_path):
    # query Image feature 뽑는 과정
    # 현재는 간단하게 MobileNetV2로 이미지의 feature를 뽑아내서 비교
    dim = 2048
    input_shape = (224, 224, 3)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False
    model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

    # 이미지 로드
    img = preprocess(image_path, input_shape)

    st = time.time()
    # 이미지에서 feature 뽑아내기
    fvec = model.predict(np.array([img]))
    print(f"[INFO] feature extract time {time.time() - st}")

    st = time.time()
    # 이미지 검색 클래스 생성
    # fvecs.bin랑 fnames.txt는 feature_extractor.py에서 만들 수 있는 파일
    imageRetrieval = ImageRetrieval(fvec_file="retrieval/fvecs.bin",
                                    fvec_img_file_name="retrieval/fvecs_names.txt",
                                    fvec_dim=dim)

    results = imageRetrieval.search(fvec)
    print(f"[INFO] image retrieval time {time.time() - st}")

    for path, dst in results:
        print(f"Image path: {path}, dst: {dst}")


if __name__ == "__main__":
    retrieval_res('image/test/image1.jpg')
