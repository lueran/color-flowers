import os

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv3D, UpSampling3D
from keras.models import Sequential


def load_image():
    x = []
    y = []
    for filename in os.listdir('./train_data/JPEG32'):
        inp = img_to_array(load_img('./train_data/JPEG32/' + filename))
        inp = np.array(inp, dtype=float)
        inp = rgb2lab(1.0 / 255 * inp)[:, :, 0]
        inp = inp.reshape(1, 32, 32, 1)
        x.append(inp)
        out = img_to_array(load_img('./train_data/JPEG96/' + filename))
        out = np.array(out, dtype=float)
        out = rgb2lab(1.0 / 255 * out)
        out = out[:, :, 1:]
        # out /= 128
        out_l = out[:, :, 0]
        out_color = out[:, :, 1:]
        out_color /= 128
        cur = np.zeros((96, 96, 3))
        cur[:, :, 0] = out[:, :, 0]
        cur[:, :, 1:] = out[:, :, 1:]/128
        # out[:, :, 0] = out_l
        # out[:, :, 1:] = out_color
        # out = out.reshape(1, 96, 96, 2)
        # out = out.reshape(1, 96, 96, 3)
        y.append(cur)
    # x = x[:500]
    # y = y[:500]
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return x, y


def build_model():
    # model = Sequential()
    # model.add(InputLayer(input_shape=(None, None, 1)))
    # model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(UpSampling2D((3, 3)))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    # model.compile(optimizer='adam', loss='mse')

    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, None, 1)))
    model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same', strides=3))
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same', strides=3))
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(UpSampling3D((3, 3, 3)))
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Conv3D(2, (3, 3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='adam', loss='mse')
    #
    # model = Sequential()
    # model.add(InputLayer(input_shape=(None, None, 1)))
    # model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    # model.compile(optimizer='adam', loss='mse')

    # model = Sequential()
    # model.add(InputLayer(input_shape=(None, None, 1)))
    # model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    # model.compile(optimizer='adam', loss='mse')
    return model


def generator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 32, 32, 1))
    batch_labels = np.zeros((batch_size, 96, 96, 3))
    while True:
        for i in range(batch_size):
            index = i
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels


def save_model(model):
    model_json = model.to_json()
    with open("./model/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./model/color_net_model.h5")


def main():
    x, y = load_image()
    model = build_model()
    print(model.output)

    # Train Model - learning rate inside 'adam' optimizer
    model.fit_generator(generator(x, y, 20), epochs=50, steps_per_epoch=1, verbose=1)

    save_model(model)


if __name__ == '__main__':
    main()