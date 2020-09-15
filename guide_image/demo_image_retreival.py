import numpy as np

import tensorflow_hub as hub
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
    base = tf.keras.applications.ResNet152(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False
    model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

    # 이미지 로드
    img = preprocess(image_path, input_shape)

    # 이미지에서 feature 뽑아내기
    fvec = model.predict(np.array([img]))

    # 이미지 검색 클래스 생성
    # fvecs.bin랑 fnames.txt는 feature_extractor.py에서 만들 수 있는 파일
    imageRetrieval = ImageRetrieval(fvec_file="guide/fvecs.bin",
                                    fvec_img_file_name="guide/fvecs_names.txt",
                                    fvec_dim=dim)

    results = imageRetrieval.search(fvec)

    for path, dst in results:
        print(f"Image path: {path}, dst: {dst}")


def retrieval_delf(image_path):
    def get_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

    img = get_image(image_path)

    delf_rt = delf(image=img,
                   score_threshold=tf.constant(100.0),
                   image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
                   max_feature_num=tf.constant(1000))

    descrip = delf_rt['descriptors']
    descrip = np.array(descrip)
    if descrip.shape[0] < 300:
        print(f"[INFO] {image_path} feature size: {descrip.shape}")
        return
    descrip = descrip[:300]
    descrip = descrip.flatten().reshape(1, -1)

    imageRetrieval = ImageRetrieval(fvec_file="guide/fvecs_delf.bin",
                                    fvec_img_file_name="guide/fvecs_delf_names.txt",
                                    fvec_dim=12000)

    results = imageRetrieval.search(descrip)
    for path, dst in results:
        print(f"Image path: {path}, dst: {dst}")


if __name__ == "__main__":
    retrieval_res('image/test/image1.jpg')
