import numpy as np

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Flatten
from tensorflow.keras import Model
import pandas as pd
import os
import cv2

tf.executing_eagerly = False

N_CLUSTER = 15

MAX_SCORE = 5

image_input = Input(shape=(224, 224, 3), name="image")
mask_input = Input(shape=(224, 224, 3), name="mask")

cnn = tf.keras.applications.MobileNetV2(include_top=False)
x = cnn(image_input)
x2 = cnn(mask_input)

# x = base_model.output
# x2 = base_model2.output

x = tf.keras.layers.BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

x2 = tf.keras.layers.BatchNormalization()(x2)
x2 = GlobalAveragePooling2D()(x2)

x = Flatten()(x)
x2 = Flatten()(x2)
x = concatenate([x, x2])
x = Dense(500, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(500, activation='relu')(x)
prediction = Dense(MAX_SCORE + 1, activation='softmax')(x)

model = Model(
    inputs=[image_input, mask_input],
    outputs=prediction
)

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_crossentropy", "acc"])

data = pd.read_csv("train.csv", header=None, names=["mask_url", "image_url", "score"])

labels = np.array(data.pop("score"), np.float32)
image_urls = np.array(data.pop("mask_url")).reshape([-1])
mask_urls = np.array(data.pop("image_url")).reshape([-1])

train_dataset = tf.data.Dataset.from_tensor_slices((mask_urls, image_urls, labels))


data = pd.read_csv("test.csv", header=None, names=["mask_url", "image_url", "score"])

labels = np.array(data.pop("score"), np.float32)
image_urls = np.array(data.pop("mask_url")).reshape([-1])
mask_urls = np.array(data.pop("image_url")).reshape([-1])

unit_size = data.shape[0] // 10

# 80%
test_size = unit_size * 8

test_dataset = tf.data.Dataset.from_tensor_slices((mask_urls, image_urls, labels))
val_dataset = test_dataset.skip(test_size)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [224, 224])


def preprocess_data(mask_url, image_url, label):
    image = tf.io.read_file(image_url)
    image = decode_img(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    mask_image = tf.io.read_file(mask_url)
    mask_image = decode_img(mask_image)
    mask_image = tf.keras.applications.mobilenet_v2.preprocess_input(mask_image)
    return {"image": image, "mask": mask_image}, label


BATCH_SIZE = 16


train_label_ds = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
val_label_ds = val_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
test_label_ds = test_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

checkpoint_path = "mobilenet_score_training/cp.ckpt"
# checkpoint_path = "efficient_net_score_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    model.load_weights(checkpoint_path)
    print('load')

# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_label_ds, batch_size=BATCH_SIZE,
          validation_data=val_label_ds,
          validation_freq=2,
          epochs=12, callbacks=[cp_callback])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_crossentropy", "acc"])

model.fit(train_label_ds, batch_size=BATCH_SIZE,
          validation_data=val_label_ds,
          validation_freq=2,
          epochs=60, callbacks=[cp_callback])


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_crossentropy", "acc"])

model.fit(train_label_ds, batch_size=BATCH_SIZE,
          validation_data=val_label_ds,
          validation_freq=2,
          epochs=100, callbacks=[cp_callback])


"""
# 예측
predictions = model.predict(test_label_ds)
for prediction in predictions:
    print(prediction, np.argmax(prediction))
"""

# 모델 평가
ev = model.evaluate(test_label_ds, batch_size=BATCH_SIZE)
print(ev)
