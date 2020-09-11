import struct
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

'''
이미지에서 특징을 뽑아서 binary 형태로 저장
'''

# 이미지 전처리 프로세싱
def preprocess(img_path, input_shape):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=input_shape[2])
    img = tf.image.resize(img, input_shape[:2])
    img = preprocess_input(img)
    return img


def extract(image_dataset, store_file="fvecs"):
    binary_file = f"{store_file}.bin"
    name_file = f"{store_file}_names.txt"

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
        for batch in dataset:
            fvecs = model.predict(batch)

            # fvecs의 길이 만큼 fmt설정
            fmt = f'{np.prod(fvecs.shape)}f'

            # fmt = 포맷, fvecs.flatten() = 벡터를 한줄로 변환
            # struct.pack을 사용하여 패킹한다.
            f.write(struct.pack(fmt, *(fvecs.flatten())))

    with open(name_file, 'w') as f:
        f.write('\n'.join(fnames))


if __name__ == '__main__':
    extract('../../image/**/*.jpg')