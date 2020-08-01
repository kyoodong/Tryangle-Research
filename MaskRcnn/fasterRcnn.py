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


BATCH_SIZE = 16
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


mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = (train_x[..., tf.newaxis] / 255.0).astype(np.float32)
test_x = (test_x[..., tf.newaxis] / 255.0).astype(np.float32)

print('train_x', train_x.shape)
print('train_y', train_y.shape)

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCH_SIZE)


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


resnet_input = Input((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), name='ResNet_Input')
x = Conv2D(64, (7, 7), (2, 2), padding='same', name='ResNet_InitialConv2D')(resnet_input)
x = MaxPool2D((3, 3), (2, 2), padding='same', name="ResNet_MaxPool")(x)
x = ResidualLayer(64, (3, 3), 256, 3)(x)
x = ResidualLayer(128, (3, 3), 512, 4)(x)
x = ResidualLayer(256, (3, 3), 1024, 23)(x)
x = ResidualLayer(512, (3, 3), 2048, 3)(x)
x = GlobalAveragePooling2D()(x)
output = Dense(10, activation='softmax')(x)


checkpoint_path = "mnist/cp.ckpt"
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
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=False)
# model.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cp_callback])
print(model.evaluate(test_ds, batch_size=BATCH_SIZE))
