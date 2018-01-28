import os
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import numpy as np
from keras.models import model_from_json


def load_model():
    # Load json and create model
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("./model/color_net_model.h5")
    print("Loaded model from disk..")
    loaded_model.compile(optimizer='adam', loss='mse')
    return loaded_model


def predict(model):
    # Load Train32
    i = 0
    for filename in os.listdir('./predict/data/'):
        image = img_to_array(load_img('./predict/data/' + filename))
        image = np.array(image, dtype=float)
        X = rgb2lab(1.0 / 255 * image)[:, :, 0]
        X = X.reshape(1, 32, 32, 1)

        # open image with Image object for resize later
        img = img_to_array(load_img('./predict/data/' + filename, target_size=(96, 96)))

        # Predict
        output = model.predict(X)
        output *= 128

        # resize black and white photo to 96X96X1
        img = np.array(img, dtype=float)
        X_96 = rgb2lab(1.0 / 255 * img)[:, :, 0]
        X_96 = X_96.reshape(1, 96, 96, 1)

        # Output colorizations
        cur = np.zeros((96, 96, 3))
        cur[:, :, 0] = X_96[0][:, :, 0]
        cur[:, :, 1:] = output[0]

        # imsave("./Results/ResultImg_"+filename+".jpg", lab2rgb(cur))
        imsave('./predict/result/' + "result_11%d.jpg" % i, lab2rgb(cur))
        i += 1


def main():
    model = load_model()
    predict(model)


if __name__ == '__main__':
    main()
