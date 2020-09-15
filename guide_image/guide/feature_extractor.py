import struct
import glob
import numpy as np
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
    return img

def extract_res(image_dataset, store_file="fvecs"):
    binary_file = f"{store_file}.bin"
    name_file = f"{store_file}_names.txt"

    batch_size = 100
    input_shape = (224, 224, 3)
    base = tf.keras.applications.ResNet152(input_shape=input_shape,
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

def get_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def extract_delf(image_dataset, store_file="fvecs_delf", extract_size=300):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print(f"[INFO] image_dataset_folder: {image_dataset}, extract_feature_size: {extract_size}")

    binary_file = f"{store_file}.bin"
    name_file = f"{store_file}_names.txt"
    print(f"[INFO] binary file: {binary_file}, name_file: {name_file}")

    fnames = glob.glob(image_dataset, recursive=True)

    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
    with open(binary_file, 'wb') as f:
        for i, image_path in enumerate(fnames):
            img = get_image(image_path)

            result = delf(image=img,
                          score_threshold=tf.constant(100.0),
                          image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
                          max_feature_num=tf.constant(1000))

            descrip = result['descriptors']
            descrip = np.array(descrip)

            if descrip.shape[0] < extract_size:
                print(f"[INFO] skipped {i} image, path: {image_path}")
                continue

            descrip = descrip[:extract_size]

            # fvecs의 길이 만큼 fmt설정
            fmt = f'{np.prod(descrip.shape)}f'

            # fmt = 포맷, fvecs.flatten() = 벡터를 한줄로 변환
            # struct.pack을 사용하여 패킹한다.
            f.write(struct.pack(fmt, *(descrip.flatten())))

            print(f"[INFO] Process {i}/{len(fnames)} images....., descrip shape: {descrip.shape}")



    with open(name_file, 'w') as f:
        f.write('\n'.join(fnames))

if __name__ == '__main__':
    extract_res('../image/**/*.jpg')
