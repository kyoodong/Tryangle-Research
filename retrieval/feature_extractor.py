import struct
import glob
import numpy as np
import os
import tensorflow_hub as hub

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

'''
이미지에서 특징을 뽑아서 binary 형태로 저장
'''

# 이미지 전처리 프로세싱
def preprocess(img_path, input_shape=None):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=input_shape[2])
    if input_shape is not None:
        img = tf.image.resize(img, input_shape[:2])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def extract(image_dataset, store_dir="features", store_file="fvecs"):
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    binary_file = f"{store_dir}/{store_file}.bin"
    name_file = f"{store_dir}/{store_file}_names.txt"

    batch_size = 100
    input_shape = (224, 224, 3)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False
    model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

    fnames = glob.glob(image_dataset, recursive=True)
    list_ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = list_ds.map(lambda x: preprocess(x, input_shape), num_parallel_calls=-1)
    dataset = ds.batch(batch_size).prefetch(-1)

    with open(binary_file, 'wb') as f:
        for i, batch in enumerate(dataset):
            fvecs = model.predict(batch)

            # fvecs의 길이 만큼 fmt설정
            fmt = f'{np.prod(fvecs.shape)}f'

            # fmt = 포맷, fvecs.flatten() = 벡터를 한줄로 변환
            # struct.pack을 사용하여 패킹한다.
            f.write(struct.pack(fmt, *(fvecs.flatten())))

            print(f"[INFO] Process {i * batch_size}/{len(fnames)} images.....")

    with open(name_file, 'w') as f:
        f.write('\n'.join(fnames))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help='Dataset path format, ex) ./image/**/*.jpg')
    parser.add_argument("--directory", required=False,
                        default="features",
                        help='Features and Image Path store File')
    parser.add_argument("--store", required=False,
                        default="fvecs",
                        help='Features and Image Path store File')
    args = parser.parse_args()

    extract(image_dataset=args.dataset,
            store_dir=args.directory,
            store_file=args.store)
