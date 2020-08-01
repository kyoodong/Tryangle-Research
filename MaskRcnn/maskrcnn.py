base_url = ''


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


BATCH_SIZE = 8
EPOCHS = 5
SUBSAMPLE_RATIO = 8
# IMAGE_SIZE = (64, 64)
IMAGE_SIZE = (28, 28)
ANCHOR_SIZES = [128, 256, 512]
ANCHOR_RATIOS = [[1,1], [math.sqrt(2), 1/math.sqrt(2)], [math.sqrt(3), 1/math.sqrt(3)]]
NUM_OF_ANCHORS = len(ANCHOR_RATIOS) * len(ANCHOR_SIZES)
SUBSAMPLED_IMAGE_SIZE = (int(IMAGE_SIZE[0] / SUBSAMPLE_RATIO), int(IMAGE_SIZE[1] / SUBSAMPLE_RATIO))
POSITIVE_THRESHOLD = 0.7
NEGATIVE_THRESHOLD = 0.3

print(tf.config.list_physical_devices('gpu'))
print(tf.config.list_logical_devices('gpu'))


# dataset 준비
train_dir = 'train/VOCdevkit/VOC2007'
test_dir = 'test/VOCdevkit/VOC2007'

def get_anchors():
    anchors = list()
    width = int(IMAGE_SIZE[0] / SUBSAMPLE_RATIO)
    height = int(IMAGE_SIZE[1] / SUBSAMPLE_RATIO)
    for x in range(width):
        for y in range(height):
            for anchor_size in ANCHOR_SIZES:
                for anchor_ratio in ANCHOR_RATIOS:
                    anchors.append([x,
                                    y,
                                    anchor_size * anchor_ratio[0],
                                    anchor_size * anchor_ratio[1]])
    return np.array(anchors)



def get_iou(inputs):
    anchors = (get_anchors() * SUBSAMPLE_RATIO).astype(np.float32)

    # intersection 영역 구하기
    intersection_left = tf.maximum(anchors[:,0], inputs[0])
    intersection_top = tf.maximum(anchors[:,1], inputs[1])
    intersection_right = tf.minimum(anchors[:,0] + anchors[:,2], inputs[0] + inputs[2])
    intersection_bottom = tf.minimum(anchors[:,1] + anchors[:,3], inputs[1] + inputs[3])
    width = tf.maximum(intersection_right - intersection_left, 0)
    height = tf.maximum(intersection_bottom - intersection_top, 0)

    # print('intersection_left', intersection_left)
    # print('intersection_top', intersection_top)
    # print('intersection_right', intersection_right)
    # print('intersection_bottom', intersection_bottom)
    # print('width', width)
    # print('height', height)

    # iou 계산
    label_area = inputs[2] * inputs[3]
    anchors_area = anchors[:,2] * anchors[:,3]
    intersection_area = width * height
    ious = intersection_area / (label_area + anchors_area - intersection_area)
    # print('label_area', label_area)
    # print('anchors_area', anchors_area)
    # print('intersection_area', intersection_area)
    # print('ious', ious)
    return ious


def get_dataset(dir, image_size):
    annotation_files = glob("{}/Annotations/0011*".format(dir))
    image_files = glob("{}/JPEGImages/0011*".format(dir))

    image_labels, bb_labels = list(), list()

    count = 0
    for annotation_file in annotation_files:
        if count % 100 == 0:
            print("{} / {}".format(count, len(annotation_files)))
        count += 1
        file = open(annotation_file, mode='r')
        file_data = file.read()
        annotation = xmltodict.parse(file_data)['annotation']
        filename = annotation['filename']
        width = float(annotation['size']['width'])
        height = float(annotation['size']['height'])
        channel = float(annotation['size']['depth'])
        im = Image.open("{}/JPEGImages/{}".format(dir, filename)).resize(image_size)
        image = np.array(im)
        image = (image / 255.0).astype(np.float32)

        def append(obj):
            name = obj['name']
            bndbox = obj['bndbox']
            x = int(bndbox['xmin'])
            y = int(bndbox['ymin'])
            bnd_width = int(bndbox['xmax']) - x
            bnd_height = int(bndbox['ymax']) - y

            # feature map size로 정규화
            x = np.float32((float(x) / width) * image_size[0])
            y = np.float32((float(y) / height) * image_size[1])
            bnd_width = np.float32((float(bnd_width) / width) * image_size[0])
            bnd_height = np.float32((float(bnd_height) / height) * image_size[1])

            image_labels.append(image)
            bb_labels.append([x, y, bnd_width, bnd_height])

            # ious = get_iou([x, y, bnd_width, bnd_height])
            # ious = tf.reshape(ious, [SUBSAMPLED_IMAGE_SIZE[0], SUBSAMPLED_IMAGE_SIZE[1], NUM_OF_ANCHORS])
            # ious = K.switch(tf.less_equal(ious, 0.3), ious - ious, ious - ious + 1)
            # ious = tf.concat([ious, 1 - ious], axis=-1)
            # obj_labels.append(ious)


        object = annotation['object']
        if isinstance(object, list):
            for obj in object:
                append(obj)
        else:
            obj = object
            append(obj)

    return np.array(image_labels), np.array(bb_labels)


# voc
# train_x, train_bb_y = get_dataset(train_dir, IMAGE_SIZE)
# test_x, test_bb_y = get_dataset(test_dir, IMAGE_SIZE)
#
# train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_bb_y)).shuffle(1000).batch(BATCH_SIZE)
# test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_bb_y)).batch(BATCH_SIZE)

#%%

dataset_dir = 'tiny-imagenet-200'
train_dir = '{}/train'.format(dataset_dir)
validation_dir = '{}/val/images'.format(dataset_dir)
test_dir = '{}/test/images'.format(dataset_dir)
words = '{}/words.txt'.format(dataset_dir)

word_bag = dict()
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

    # for label in labels:
    #     if label not in word_bag:
    #         word_bag[label] = 1
    #         word_bag_list.append(label)
    word_bag_list.append(directory)
    # word_map[directory] = [word_bag[label] for label in labels]

print('category count = ', len(word_bag_list))

word_bag_list = np.array(word_bag_list)
word_file.close()

# Get train set
# train_dir_list = glob(train_dir)
# for dir in train_dir_list:
#     image_dir = '{}/images'.format(dir)
#     images = glob(image_dir)


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
# train_ds = tf.data.Dataset.list_files('{}/n01443537/images/*'.format(train_dir)).shuffle(1000)
train_ds = train_ds.map(process_path, num_parallel_calls=BATCH_SIZE).batch(BATCH_SIZE)

valid_image_paths = glob("{}/*".format(validation_dir))
valid_annotation_path = "{}/{}/val_annotations.txt".format(dataset_dir, 'val')
valid_images = []
valid_image_labels = []

# for valid_image_path in valid_image_paths:
#     image = Image.open(valid_image_path)
#     image = np.array(image)
#     image = (image / 255.0).astype(np.float32)
#     valid_images.append(image)

val_label_dict = dict()
valid_annotation_file = open(valid_annotation_path, 'r')
while True:
    line = valid_annotation_file.readline()
    if not line:
        break
    datas = line.split('\t')
    val_label_dict[datas[0]] = datas[1]
    valid_image_labels.append(np.cast['float32'](datas[1] == word_bag_list))


valid_image_labels = np.array(valid_image_labels)
# print(valid_image_labels.shape, valid_images.shape)
# valid_ds = tf.data.Dataset.from_tensor_slices([valid_images, valid_image_labels]).batch(BATCH_SIZE)
valid_annotation_file.close()

# valid_ds
# print(valid_ds.take(0))

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, validation_split=0.2,
#                                                                subset='training', seed=123,
#                                                                image_size=IMAGE_SIZE,
#                                                                batch_size=BATCH_SIZE)

# class_name = train_ds.class_names


def process_valid_path(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.image.resize(image, IMAGE_SIZE)


valid_ds = tf.data.Dataset.list_files('{}/*'.format(validation_dir))
valid_ds = valid_ds.map(process_valid_path)
valid_label_ds = tf.data.Dataset.from_tensor_slices([valid_image_labels]).unbatch()
valid_ds = tf.data.Dataset.zip((valid_ds, valid_label_ds)).batch(BATCH_SIZE)


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

# class RPN(Model):
#     def __init__(self):
#         super(RPN, self).__init__()
#
#     def get_anchor(self, inputs):
#         anchor_list = list()
#         for anchor_size in ANCHOR_SIZES:
#             for anchor_ratio in ANCHOR_RATIOS:
#                 anchor_list.append()
#
#     def sliding_window(self, inputs):
#         for x in range(SUBSAMPLED_IMAGE_SIZE[0]):
#             for y in range(SUBSAMPLED_IMAGE_SIZE[1]):
#                 window = tf.slice(inputs, (x, y, 0), (3, 3, 256))
#                 self.get_anchor(window)
#
#     def call(self, inputs, training=None, mask=None):
#         print("inputs = {}".format(inputs))
#         inputs = tf.map_fn(self.sliding_window, inputs)

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


checkpoint_path = "tiny_imagenet/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


def loss(y_true, y_pred):
    print('y_true', y_true)
    print('y_pred', y_pred)
    return K.categorical_crossentropy(y_true, y_pred)


model = Model(resnet_input, output)
model.load_weights(checkpoint_path)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cp_callback],
          validation_data=valid_ds, validation_batch_size=BATCH_SIZE, validation_freq=int(EPOCHS / 5))
