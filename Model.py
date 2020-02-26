import os
from tensorflow.keras import layers, models, callbacks, Sequential
import tensorflow as tf
import numpy as np


def clean_result():
    if os.path.exists("./logs"):
        os.system("rm -rf ./logs/")
    else:
        os.mkdir("./logs")
    return None


def getInception(one, threeReduce, three, fiveReduce, five, pool, input):
    inc_conv_1x1_1_0 = layers.Conv2D(one, (1, 1), activation='relu')(input)
    inc_conv_1x1_1_1 = layers.Conv2D(threeReduce, (1, 1), activation='relu')(input)
    inc_conv_3x3_1_1 = layers.Conv2D(three, (3, 3), padding='same', activation='relu')(inc_conv_1x1_1_1)
    inc_conv_1x1_1_2 = layers.Conv2D(fiveReduce, (1, 1), activation='relu')(input)
    inc_conv_5x5_1_2 = layers.Conv2D(five, (5, 5), padding='same', activation='relu')(inc_conv_1x1_1_2)
    inc_max_3x3_1_3 = layers.MaxPooling2D((3, 3), padding='same', strides=1)(input)
    inc_conv_1x1_1_3 = layers.Conv2D(pool, (1, 1), activation='relu')(inc_max_3x3_1_3)
    return layers.concatenate([
        inc_conv_1x1_1_0,
        inc_conv_3x3_1_1,
        inc_conv_5x5_1_2,
        inc_conv_1x1_1_3
    ])


def google_net(width, height):
    normalizationEpsilon = 1e-6
    input = layers.Input(shape=(width, height, 3))
    conv_7x7_2_1 = layers.Conv2D(64, (7, 7), activation='relu', strides=2, padding='same')(input)
    max_3x3_2_1 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(conv_7x7_2_1)
    norm_1 = layers.LayerNormalization(epsilon=normalizationEpsilon)(max_3x3_2_1)
    conv_1x1_1_2 = layers.Conv2D(64, (1, 1), activation='relu')(norm_1)
    conv_3x3_1_2 = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(conv_1x1_1_2)
    norm_2 = layers.LayerNormalization(epsilon=normalizationEpsilon)(conv_3x3_1_2)
    max_3x3_2_2 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(norm_2)
    inc_3a = getInception(64, 96, 128, 16, 32, 32, max_3x3_2_2)
    inc_3b = getInception(128, 128, 192, 32, 96, 64, inc_3a)
    max_3x3_2_3 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(inc_3b)
    inc_4a = getInception(192, 96, 208, 16, 48, 64, max_3x3_2_3)
    inc_4b = getInception(160, 112, 224, 24, 64, 64, inc_4a)
    inc_4c = getInception(128, 128, 256, 24, 64, 64, inc_4b)
    inc_4d = getInception(112, 144, 288, 32, 64, 64, inc_4c)
    inc_4e = getInception(256, 160, 320, 32, 128, 128, inc_4d)
    max_3x3_2_4 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(inc_4e)
    inc_5a = getInception(256, 160, 320, 32, 128, 128, max_3x3_2_4)
    inc_5b = getInception(384, 192, 384, 48, 128, 128, inc_5a)
    avg_6 = layers.AveragePooling2D((7, 7), padding='same')(inc_5b)
    dropout_6 = layers.Dropout(0.4)(max_3x3_2_1)
    flatten = layers.Flatten()(dropout_6)
    fc_6 = layers.Dense(1000, activation='relu')(flatten)
    dropout_7 = layers.Dropout(0.4)(fc_6)
    fc_7 = layers.Dense(10, activation='softmax')(dropout_7)
    model = tf.keras.Model(inputs=input, outputs=fc_7, name='google_net')
    return model


def AlexNet(width, height):
    model = Sequential(name='AlexNet')
    model.add(layers.Conv2D(96, (11, 11), strides=(2, 2), input_shape=(width, height, 3),
                            padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(
        layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(
        layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(
        layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model



def BP(width, height):
    model = Sequential(name='BP')
    model.add(layers.Dense(32, input_shape=(width, height, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    return model


def SeNet(width, height, num_rate=16):
    input = layers.Input(shape=(width, height, 3))
    x = tf.keras.applications.InceptionV3(include_top=False,
                                          weights='imagenet',
                                          input_tensor=None,
                                          input_shape=(width, height, 3),
                                          pooling=max)(input)
    squeeze = layers.GlobalAveragePooling2D()(x)
    excitation = layers.Dense(units=2048 // num_rate, activation='relu')(squeeze)
    excitation = layers.Dense(units=2048, activation='relu')(excitation)
    excitation = layers.Reshape((1, 1, 2048))(excitation)
    scale = layers.multiply([x, excitation])
    x = layers.GlobalAveragePooling2D()(scale)
    fc = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=input, outputs=fc, name='senet')
    return model
