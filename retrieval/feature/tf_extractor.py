import struct
import numpy as np
import os
import glob

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

INPUT_SIZE = (224, 224, 3)
DIMENTION = 1280
base = tf.keras.applications.MobileNetV2(input_shape=INPUT_SIZE,
                                         include_top=False,
                                         weights='imagenet')
base.trainable = False
MODEL = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))


# 이미지 전처리 프로세싱
def preprocess(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=INPUT_SIZE[2])
    if INPUT_SIZE is not None:
        img = tf.image.resize(img, INPUT_SIZE[:2])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def extract(image_dataset, store_dir="output", store_file="fvecs"):
    """
    이미지들 feature를 뽑아냄
    
    :param image_dataset: feature를 추출할 이미지 디렉토리
    :param store_dir: 이미지 feature를 저장할 디렉토리
    :param store_file: 이미지 feature 파일의 이름
    """

    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    binary_file = f"{store_dir}/{store_file}.bin"
    name_file = f"{store_dir}/{store_file}_names.txt"
    image_dataset = f"{image_dataset}/**/*.jpg"

    batch_size = 100

    fnames = glob.glob(image_dataset, recursive=True)

    list_ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = list_ds.map(lambda x: preprocess(x), num_parallel_calls=-1)
    dataset = ds.batch(batch_size).prefetch(-1)

    size = len(fnames)
    processed = 0
    with open(binary_file, 'wb') as f:
        for i, batch in enumerate(dataset):
            fvecs = MODEL.predict(batch)

            # fvecs의 길이 만큼 fmt설정
            fmt = f'{np.prod(fvecs.shape)}f'

            # fmt = 포맷, fvecs.flatten() = 벡터를 한줄로 변환
            # struct.pack을 사용하여 패킹한다.
            f.write(struct.pack(fmt, *(fvecs.flatten())))

            processed += min(batch_size, max(size - batch_size * i, 0))
            print(f"[INFO] Process {processed}/{size} images.....")


    with open(name_file, 'w') as f:
        f.write('\n'.join(fnames))


def extract_individual(img,
                       image_name,
                       output_dir="output"):
    """
    
    :param img: 이미지 numpy 배열
    :param image_name: 이미지 이름
    :param output_dir: 이미지이름.feature 파일이 저장될 디렉토리
    """
    def preprocess_individual(img_numpy):
        h, w, c = img_numpy.shape
        img = np.reshape(img_numpy, (1, h, w, c))
        img = tf.image.resize(img, INPUT_SIZE[:2])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img

    img = preprocess_individual(img)
    try:
        with open(os.path.join(output_dir, f"{image_name}.feature"), "wb") as f:
            fvecs = MODEL.predict(img)

            # fvecs의 길이 만큼 fmt설정
            fmt = f'{np.prod(fvecs.shape)}f'

            # fmt = 포맷, fvecs.flatten() = 벡터를 한줄로 변환
            # struct.pack을 사용하여 패킹한다.
            f.write(struct.pack(fmt, *(fvecs.flatten())))
    except Exception as e:
        print(e)
        return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help='Dataset directory path')
    parser.add_argument("--directory", required=False,
                        default="output",
                        help='Features and Image Path store directory')
    parser.add_argument("--store", required=False,
                        default="fvecs",
                        help='Features and Image Path store File')
    args = parser.parse_args()

    extract(image_dataset=args.dataset,
            store_dir=args.directory,
            store_file=args.store)

