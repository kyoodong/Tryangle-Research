#%%

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, Activation, MaxPool2D, Dense, Input, Reshape, Activation
import tensorflow.keras.backend as K
from glob import glob
import math
import xmltodict
from PIL import Image, ImageDraw
import os
import sys

EPOCHS = 12
BATCH_SIZE = 8
SUBSAMPLE_RATIO = 4
IMAGE_SIZE = (64, 64)
ANCHOR_SIZES = [128, 256, 512]
ANCHOR_RATIOS = [[1,1], [math.sqrt(2), 1/math.sqrt(2)], [math.sqrt(3), 1/math.sqrt(3)]]
NUM_OF_ANCHORS = len(ANCHOR_RATIOS) * len(ANCHOR_SIZES)
SUBSAMPLED_IMAGE_SIZE = (int(IMAGE_SIZE[0] / SUBSAMPLE_RATIO), int(IMAGE_SIZE[1] / SUBSAMPLE_RATIO))
POSITIVE_THRESHOLD = 0.7
NEGATIVE_THRESHOLD = 0.3

dataset_dir = 'tiny-imagenet-200'
train_dir = '{}/train'.format(dataset_dir)
validation_dir = '{}/val/images'.format(dataset_dir)
test_dir = '{}/test/images'.format(dataset_dir)
words = '{}/words.txt'.format(dataset_dir)

word_bag_list = list()
word_map = dict()
word_file = open(words, 'r')
while True:
    line = word_file.readline()
    if not line:
        break
    data = line.split('\t')
    directory = data[0]
    labels = data[1].replace('\n', '').replace(' ', '').split(',')
    word_bag_list.append(directory)

word_bag_list = np.array(word_bag_list)
word_file.close()


def get_label(path):
    part = tf.strings.split(path, os.path.sep)
    dir_name = part[2]
    one_hot = word_bag_list == dir_name
    one_hot = tf.cast(one_hot, tf.float32)
    return one_hot


def get_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.image.resize(image, IMAGE_SIZE)


def process_path(path):
    image = get_image(path)
    label = get_label(path)
    return image, label


train_ds = tf.data.Dataset.list_files('{}/*/images/*'.format(train_dir)).shuffle(1000)
train_ds = train_ds.map(process_path, num_parallel_calls=BATCH_SIZE).batch(BATCH_SIZE)

valid_image_paths = glob("{}/*".format(validation_dir))
valid_annotation_path = "{}/{}/val_annotations.txt".format(dataset_dir, 'val')
valid_images = []

val_keys = []
val_values = []
valid_annotation_file = open(valid_annotation_path, 'r')
while True:
    line = valid_annotation_file.readline()
    if not line:
        break
    datas = line.split('\t')
    val_keys.append(datas[0])
    val_values.append(datas[1])

valid_annotation_file.close()

table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_keys, val_values, key_dtype=tf.string, value_dtype=tf.string), "null"
)


def process_valid_path(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32) / 255.0

    part = tf.strings.split(path, os.path.sep)
    filename = part[-1]
    one_hot = word_bag_list == table.lookup(filename)
    one_hot = tf.cast(one_hot, tf.float32)
    return tf.image.resize(image, IMAGE_SIZE), one_hot


valid_ds = tf.data.Dataset.list_files('{}/*'.format(validation_dir))
valid_ds = valid_ds.map(process_valid_path).batch(BATCH_SIZE)
# valid_label_ds = tf.data.Dataset.from_tensor_slices([valid_image_labels]).unbatch()
# valid_ds = tf.data.Dataset.zip((valid_ds, valid_label_ds)).batch(BATCH_SIZE)


#%%

class ResidualUnit(Model):
    def __init__(self, shrank_filter_size, kernel_size, filter_size):
        super(ResidualUnit, self).__init__()
        self.identifier = Conv2D(filter_size, (1, 1))
        self.downsampling = Conv2D(shrank_filter_size, (1, 1))
        self.conv = Conv2D(shrank_filter_size, kernel_size, padding='same')
        self.upsampling = Conv2D(filter_size, (1, 1))
        self.bn = BatchNormalization()
        self.relu = Activation('relu')

    def call(self, inputs, training=None, mask=None):
        x = self.downsampling(inputs)
        x = self.conv(x)
        x = self.upsampling(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return self.identifier(inputs) + x


class ResidualLayer(Model):
    def __init__(self, shrank_filter_size, kernel_size, filter_size, iter_count):
        super(ResidualLayer, self).__init__()

        self.layer_list = list()
        for _ in range(iter_count):
            self.layer_list.append(ResidualUnit(shrank_filter_size, kernel_size, filter_size))

    def call(self, inputs, training=None, mask=None):
        for layer in self.layer_list:
            inputs = layer(inputs, training=training)
        return inputs

#%%

# ResNet
resnet_input = Input((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name='ResNet_Input')
x = Conv2D(64, (7, 7), (2, 2), padding='same', name='ResNet_InitialConv2D')(resnet_input)
x = MaxPool2D((3, 3), (2, 2), padding='same', name="ResNet_MaxPool")(x)
x = ResidualLayer(64, (3, 3), 256, 3)(x)
x = ResidualLayer(128, (3, 3), 512, 4)(x)
x = ResidualLayer(256, (3, 3), 1024, 23)(x)
x = ResidualLayer(512, (3, 3), 2048, 3)(x)
x = GlobalAveragePooling2D()(x)
output = Dense(len(word_bag_list), activation='softmax')(x)


checkpoint_path = "tiny_imagenet_t/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


def loss(y_true, y_pred):
    print('loss y_true', y_true)
    print(tf.math.count_nonzero(y_true))
    print('loss y_pred', y_pred)
    print(tf.math.count_nonzero(y_pred))
    return K.categorical_crossentropy(y_true, y_pred)

def metrics(y_true, y_pred):
    print('metrics y_true', y_true)
    print(tf.math.count_nonzero(y_true))
    print('metrics y_pred', y_pred)
    print(tf.math.count_nonzero(y_pred))
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)


model = Model(resnet_input, output)
model.load_weights(checkpoint_path)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cp_callback], validation_data=valid_ds, validation_freq=4)
model.evaluate(valid_ds, batch_size=BATCH_SIZE)
