import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras import Model
import pandas as pd
import os

tf.executing_eagerly = False

N_CLUSTER = 15
MAX_SCORE = 5

image_input = Input(shape=(None, None, 3), name="image")
cluster_input = Input(shape=(1, ), name="cluster")

# preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input(image_input)
base_model = tf.keras.applications.MobileNetV2(include_top=False, input_tensor=image_input)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = concatenate([x, cluster_input])
x = Dense(1024, activation='relu')(x)
prediction = Dense(MAX_SCORE + 1, activation='softmax')(x)

model = Model(
    inputs=[image_input, cluster_input],
    outputs=prediction
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_crossentropy", "acc"])

data = pd.read_csv("data.csv", header=None, names=["url", "cluster", "score"])

clusters = np.array(data.pop("cluster"), np.float32)
labels = np.array(data.pop("score"), np.float32)
urls = np.array(data).reshape([-1])

unit_size = data.shape[0] // 10

# 80%
train_size = unit_size * 8

# 10%
test_size = unit_size

full_dataset = tf.data.Dataset.from_tensor_slices((urls, clusters, labels))
full_dataset = full_dataset.shuffle(20000)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)


normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [224, 224])


def preprocess_data(url, cluster, label):
    image = tf.io.read_file(url)
    image = decode_img(image)
    image = normalization_layer(image)
    return {"image": image, "cluster": cluster}, label


BATCH_SIZE = 32
train_label_ds = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
test_label_ds = test_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
val_label_ds = val_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

checkpoint_path = "score_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    model.load_weights(checkpoint_path)
    print('load')

# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
HOUR = 6

model.fit(train_label_ds, batch_size=BATCH_SIZE,
          validation_data=val_label_ds,
          validation_freq=5,
          epochs=HOUR * 8, callbacks=[cp_callback])

predictions = model.predict(test_label_ds)
for prediction in predictions:
    print(prediction, np.argmax(prediction))
