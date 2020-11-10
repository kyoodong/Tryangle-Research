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

nose_x_input = Input(shape=(1, ), name="nose_x")
nose_y_input = Input(shape=(1, ), name="nose_y")
left_eye_x_input = Input(shape=(1, ), name="left_eye_x")
left_eye_y_input = Input(shape=(1, ), name="left_eye_y")
right_eye_x_input = Input(shape=(1, ), name="right_eye_x")
right_eye_y_input = Input(shape=(1, ), name="right_eye_y")
left_ear_x_input = Input(shape=(1, ), name="left_ear_x")
left_ear_y_input = Input(shape=(1, ), name="left_ear_y")
right_ear_x_input = Input(shape=(1, ), name="right_ear_x")
right_ear_y_input = Input(shape=(1, ), name="right_ear_y")
left_shoulder_x_input = Input(shape=(1, ), name="left_shoulder_x")
left_shoulder_y_input = Input(shape=(1, ), name="left_shoulder_y")
right_shoulder_x_input = Input(shape=(1, ), name="right_shoulder_x")
right_shoulder_y_input = Input(shape=(1, ), name="right_shoulder_y")
left_elbow_x_input = Input(shape=(1, ), name="left_elbow_x")
left_elbow_y_input = Input(shape=(1, ), name="left_elbow_y")
right_elbow_x_input = Input(shape=(1, ), name="right_elbow_x")
right_elbow_y_input = Input(shape=(1, ), name="right_elbow_y")
left_wrist_x_input = Input(shape=(1, ), name="left_wrist_x")
left_wrist_y_input = Input(shape=(1, ), name="left_wrist_y")
right_wrist_x_input = Input(shape=(1, ), name="right_wrist_x")
right_wrist_y_input = Input(shape=(1, ), name="right_wrist_y")
left_hip_x_input = Input(shape=(1, ), name="left_hip_x")
left_hip_y_input = Input(shape=(1, ), name="left_hip_y")
right_hip_x_input = Input(shape=(1, ), name="right_hip_x")
right_hip_y_input = Input(shape=(1, ), name="right_hip_y")
left_knee_x_input = Input(shape=(1, ), name="left_knee_x")
left_knee_y_input = Input(shape=(1, ), name="left_knee_y")
right_knee_x_input = Input(shape=(1, ), name="right_knee_x")
right_knee_y_input = Input(shape=(1, ), name="right_knee_y")
left_ankle_x_input = Input(shape=(1, ), name="left_ankle_x")
left_ankle_y_input = Input(shape=(1, ), name="left_ankle_y")
right_ankle_x_input = Input(shape=(1, ), name="right_ankle_x")
right_ankle_y_input = Input(shape=(1, ), name="right_ankle_y")


base_model = tf.keras.applications.ResNet101(include_top=False, input_tensor=image_input)
# base_model = tf.keras.applications.EfficientNetB7(include_top=False, input_tensor=image_input)

x = base_model.output
x = tf.keras.layers.BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = concatenate([x, nose_x_input, nose_y_input, left_eye_x_input, left_eye_y_input, right_eye_x_input, right_eye_y_input,
     left_ear_x_input, left_ear_y_input, right_ear_x_input, right_ear_y_input,
     left_shoulder_x_input, left_shoulder_y_input, right_shoulder_x_input, right_shoulder_y_input,
     left_elbow_x_input, left_elbow_y_input, right_elbow_x_input, right_elbow_y_input,
     left_wrist_x_input, left_wrist_y_input, right_wrist_x_input, right_wrist_y_input,
     left_hip_x_input, left_hip_y_input, right_hip_x_input, right_hip_y_input,
     left_knee_x_input, left_knee_y_input, right_knee_x_input, right_knee_y_input,
     left_ankle_x_input, left_ankle_y_input, right_ankle_x_input, right_ankle_y_input])
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(MAX_SCORE + 1, activation='softmax')(x)

model = Model(
    inputs=[
        image_input],
    outputs=prediction
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_crossentropy", "acc"])

data = pd.read_csv("data.csv", header=None, names=["url", "id", "object_id", "pose_id", "nose_x", "nose_y",
                                                   "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
                                                   "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y",
                                                   "left_shoulder_x", "left_shoulder_y", "right_shoulder_x", "right_shoulder_y",
                                                   "left_elbow_x", "left_elbow_y", "right_elbow_x", "right_elbow_y",
                                                   "left_wrist_x", "left_wrist_y", "right_wrist_x", "right_wrist_y",
                                                   "left_hip_x", "left_hip_y", "right_hip_x", "right_hip_y",
                                                   "left_knee_x", "left_knee_y", "right_knee_x", "right_knee_y",
                                                   "left_ankle_x", "left_ankle_y", "right_ankle_x", "right_ankle_y",
                                                   "score"])

id = np.array(data.pop("id"), np.float32)
object_id = np.array(data.pop("object_id"), np.float32)
pose_id = np.array(data.pop("pose_id"), np.float32)
nose_x = np.array(data.pop("nose_x"), np.float32)
nose_y = np.array(data.pop("nose_y"), np.float32)
left_eye_x = np.array(data.pop("left_eye_x"), np.float32)
left_eye_y = np.array(data.pop("left_eye_y"), np.float32)
right_eye_x = np.array(data.pop("right_eye_x"), np.float32)
right_eye_y = np.array(data.pop("right_eye_y"), np.float32)
left_ear_x = np.array(data.pop("left_ear_x"), np.float32)
left_ear_y = np.array(data.pop("left_ear_y"), np.float32)
right_ear_x = np.array(data.pop("right_ear_x"), np.float32)
right_ear_y = np.array(data.pop("right_ear_y"), np.float32)
left_shoulder_x = np.array(data.pop("left_shoulder_x"), np.float32)
left_shoulder_y = np.array(data.pop("left_shoulder_y"), np.float32)
right_shoulder_x = np.array(data.pop("right_shoulder_x"), np.float32)
right_shoulder_y = np.array(data.pop("right_shoulder_y"), np.float32)
left_elbow_x = np.array(data.pop("left_elbow_x"), np.float32)
left_elbow_y = np.array(data.pop("left_elbow_y"), np.float32)
right_elbow_x = np.array(data.pop("right_elbow_x"), np.float32)
right_elbow_y = np.array(data.pop("right_elbow_y"), np.float32)
left_wrist_x = np.array(data.pop("left_wrist_x"), np.float32)
left_wrist_y = np.array(data.pop("left_wrist_y"), np.float32)
right_wrist_x = np.array(data.pop("right_wrist_x"), np.float32)
right_wrist_y = np.array(data.pop("right_wrist_y"), np.float32)
left_hip_x = np.array(data.pop("left_hip_x"), np.float32)
left_hip_y = np.array(data.pop("left_hip_y"), np.float32)
right_hip_x = np.array(data.pop("right_hip_x"), np.float32)
right_hip_y = np.array(data.pop("right_hip_y"), np.float32)
left_knee_x = np.array(data.pop("left_knee_x"), np.float32)
left_knee_y = np.array(data.pop("left_knee_y"), np.float32)
right_knee_x = np.array(data.pop("right_knee_x"), np.float32)
right_knee_y = np.array(data.pop("right_knee_y"), np.float32)
left_ankle_x = np.array(data.pop("left_ankle_x"), np.float32)
left_ankle_y = np.array(data.pop("left_ankle_y"), np.float32)
right_ankle_x = np.array(data.pop("right_ankle_x"), np.float32)
right_ankle_y = np.array(data.pop("right_ankle_y"), np.float32)
labels = np.array(data.pop("score"), np.float32)
urls = np.array(data).reshape([-1])

train_dataset = tf.data.Dataset.from_tensor_slices(
    (urls, nose_x, nose_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
     left_ear_x, left_ear_y, right_ear_x, right_ear_y,
     left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y,
     left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y,
     left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y,
     left_hip_x, left_hip_y, right_hip_x, right_hip_y,
     left_knee_x, left_knee_y, right_knee_x, right_knee_y,
     left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, labels))


data = pd.read_csv("test_data.csv", header=None, names=["url", "id", "object_id", "pose_id", "nose_x", "nose_y",
                                                   "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
                                                   "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y",
                                                   "left_shoulder_x", "left_shoulder_y", "right_shoulder_x", "right_shoulder_y",
                                                   "left_elbow_x", "left_elbow_y", "right_elbow_x", "right_elbow_y",
                                                   "left_wrist_x", "left_wrist_y", "right_wrist_x", "right_wrist_y",
                                                   "left_hip_x", "left_hip_y", "right_hip_x", "right_hip_y",
                                                   "left_knee_x", "left_knee_y", "right_knee_x", "right_knee_y",
                                                   "left_ankle_x", "left_ankle_y", "right_ankle_x", "right_ankle_y",
                                                   "score"])

id = np.array(data.pop("id"), np.float32)
object_id = np.array(data.pop("object_id"), np.float32)
pose_id = np.array(data.pop("pose_id"), np.float32)
nose_x = np.array(data.pop("nose_x"), np.float32)
nose_y = np.array(data.pop("nose_y"), np.float32)
left_eye_x = np.array(data.pop("left_eye_x"), np.float32)
left_eye_y = np.array(data.pop("left_eye_y"), np.float32)
right_eye_x = np.array(data.pop("right_eye_x"), np.float32)
right_eye_y = np.array(data.pop("right_eye_y"), np.float32)
left_ear_x = np.array(data.pop("left_ear_x"), np.float32)
left_ear_y = np.array(data.pop("left_ear_y"), np.float32)
right_ear_x = np.array(data.pop("right_ear_x"), np.float32)
right_ear_y = np.array(data.pop("right_ear_y"), np.float32)
left_shoulder_x = np.array(data.pop("left_shoulder_x"), np.float32)
left_shoulder_y = np.array(data.pop("left_shoulder_y"), np.float32)
right_shoulder_x = np.array(data.pop("right_shoulder_x"), np.float32)
right_shoulder_y = np.array(data.pop("right_shoulder_y"), np.float32)
left_elbow_x = np.array(data.pop("left_elbow_x"), np.float32)
left_elbow_y = np.array(data.pop("left_elbow_y"), np.float32)
right_elbow_x = np.array(data.pop("right_elbow_x"), np.float32)
right_elbow_y = np.array(data.pop("right_elbow_y"), np.float32)
left_wrist_x = np.array(data.pop("left_wrist_x"), np.float32)
left_wrist_y = np.array(data.pop("left_wrist_y"), np.float32)
right_wrist_x = np.array(data.pop("right_wrist_x"), np.float32)
right_wrist_y = np.array(data.pop("right_wrist_y"), np.float32)
left_hip_x = np.array(data.pop("left_hip_x"), np.float32)
left_hip_y = np.array(data.pop("left_hip_y"), np.float32)
right_hip_x = np.array(data.pop("right_hip_x"), np.float32)
right_hip_y = np.array(data.pop("right_hip_y"), np.float32)
left_knee_x = np.array(data.pop("left_knee_x"), np.float32)
left_knee_y = np.array(data.pop("left_knee_y"), np.float32)
right_knee_x = np.array(data.pop("right_knee_x"), np.float32)
right_knee_y = np.array(data.pop("right_knee_y"), np.float32)
left_ankle_x = np.array(data.pop("left_ankle_x"), np.float32)
left_ankle_y = np.array(data.pop("left_ankle_y"), np.float32)
right_ankle_x = np.array(data.pop("right_ankle_x"), np.float32)
right_ankle_y = np.array(data.pop("right_ankle_y"), np.float32)
labels = np.array(data.pop("score"), np.float32)
urls = np.array(data).reshape([-1])

unit_size = data.shape[0] // 10

# 80%
test_size = unit_size * 8

test_dataset = tf.data.Dataset.from_tensor_slices(
    (urls, nose_x, nose_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
     left_ear_x, left_ear_y, right_ear_x, right_ear_y,
     left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y,
     left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y,
     left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y,
     left_hip_x, left_hip_y, right_hip_x, right_hip_y,
     left_knee_x, left_knee_y, right_knee_x, right_knee_y,
     left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, labels)
)
val_dataset = test_dataset.skip(test_size)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [224, 224])


def preprocess_data(url, nose_x, nose_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
     left_ear_x, left_ear_y, right_ear_x, right_ear_y,
     left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y,
     left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y,
     left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y,
     left_hip_x, left_hip_y, right_hip_x, right_hip_y,
     left_knee_x, left_knee_y, right_knee_x, right_knee_y,
     left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, label):
    image = tf.io.read_file(url)
    image = decode_img(image)
    image = normalization_layer(image)
    return {"image": image, "nose_x": nose_x, "nose_y": nose_y, "left_eye_x": left_eye_x, "left_eye_y": left_eye_y,
            "right_eye_x": right_eye_x, "right_eye_y": right_eye_y,
            "left_ear_x": left_ear_x, "left_ear_y": left_ear_y,
            "right_ear_x": right_ear_x, "right_ear_y": right_ear_y,
            "left_shoulder_x": left_shoulder_x, "left_shoulder_y": left_shoulder_y,
            "right_shoulder_x": right_shoulder_x, "right_shoulder_y": right_shoulder_y,
            "left_elbow_x": left_elbow_x, "left_elbow_y": left_elbow_y,
            "right_elbow_x": right_elbow_x, "right_elbow_y": right_elbow_y,
            "left_wrist_x": left_wrist_x, "left_wrist_y": left_wrist_y,
            "right_wrist_x": right_wrist_x, "right_wrist_y": right_wrist_y,
            "left_hip_x": left_hip_x, "left_hip_y": left_hip_y,
            "right_hip_x": right_hip_x, "right_hip_y": right_hip_y,
            "left_knee_x": left_knee_x, "left_knee_y": left_knee_y,
            "right_knee_x": right_knee_x, "right_knee_y": right_knee_y,
            "left_ankle_x": left_ankle_x, "left_ankle_y": left_ankle_y,
            "right_ankle_x": right_ankle_x, "right_ankle_y": right_ankle_y}, label


BATCH_SIZE = 4


train_label_ds = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
val_label_ds = val_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
test_label_ds = test_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

checkpoint_path = "resnet_152_score_training/cp.ckpt"
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
          validation_freq=5,
          epochs=200, callbacks=[cp_callback])


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_crossentropy", "acc"])

model.fit(train_label_ds, batch_size=BATCH_SIZE,
          validation_data=val_label_ds,
          validation_freq=5,
          epochs=100
          , callbacks=[cp_callback])


"""
# 예측
predictions = model.predict(test_label_ds)
for prediction in predictions:
    print(prediction, np.argmax(prediction))
"""

# 모델 평가
ev = model.evaluate(test_label_ds, batch_size=BATCH_SIZE)
print(ev)

