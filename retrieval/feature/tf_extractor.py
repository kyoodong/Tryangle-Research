import struct
import numpy as np
import os

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

def preprocess(img_numpy):
    h, w, c = img_numpy.shape
    img = np.reshape(img_numpy, (1, h, w, c))
    img = tf.image.resize(img, INPUT_SIZE[:2])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


def extract_individual(img,
                    image_name,
                    output_dir="output"):
    img = preprocess(img)
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

    import cv2
    image_dir = "../../images"
    image_names = [f"test{i}.jpg"for i in range(1, 27)]

    for img_name in image_names:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        b = extract_individual(img, img_name)
        if b:
            print(f"Success {img_name}")
        else:
            print(f"Failed {img_name}")



