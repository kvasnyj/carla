import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Dropout, ELU, Lambda
from keras.models import Sequential
from keras.utils import np_utils

def image_pipeline(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img =  img[:,:, np.newaxis]
    return img

def get_data():
    X = []
    y = []

    for file in glob.glob('data/vehicles/**/*.png', recursive=True):
        X.append(image_pipeline(file))
        y.append(1)

    for file in glob.glob('data/non-vehicles/**/*.png', recursive=True):
        X.append(image_pipeline(file))
        y.append(0)

    y = np_utils.to_categorical(y, 2)

    return train_test_split(np.array(X), np.array(y), test_size=0.2)

def get_model_nvidia(img): #source of the model: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    shape = img.shape
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=shape, output_shape=shape))
    model.add(Conv2D(3, 5, 5,  subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    model.add(Flatten())
    model.add(Dropout(.2))

    model.add(Dense(1164))
    model.add(Dropout(.2))
    model.add(ELU())

    model.add(Dense(100))
    model.add(Dropout(.2))
    model.add(ELU())

    model.add(Dense(50))
    model.add(Dropout(.2))
    model.add(ELU())

    model.add(Dense(10))
    model.add(Dropout(.2))
    model.add(ELU())

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer="adam", loss="mse")

    model.summary()

    return model    

X_train, X_test, y_train, y_test =  get_data() 
n_train = len(X_train)
n_test = len(X_test)
n = n_train + n_test

print("size: %s, train: %s, test: %s" % (n, n_train, n_test))

model = get_model_nvidia(X_train[0])

from keras.preprocessing.image import ImageDataGenerator # data augmentation
datagen = ImageDataGenerator(
            width_shift_range=0.1, 
            height_shift_range=0.1,
            vertical_flip = True,
            rescale = 1.2) 
datagen.fit(X_train)
# model.fit(x = X_train, y = y_train, nb_epoch = 10, batch_size = 100, validation_split=0.2, shuffle=True, verbose=0)
model.fit_generator(datagen.flow(X_train, y_train, batch_size = 100), nb_epoch = 20, samples_per_epoch=X_train.shape[0], verbose=0)

loss = model.evaluate(X_test, y_test, verbose=0) 
print(loss)

import json
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
model.save_weights('model.h5')

val = []
val.append(image_pipeline('./data/non-vehicles/Extras/extra45.png')) 
val.append(image_pipeline('./data/vehicles/GTI_Left/image0041.png')) 

prediction = model.predict(np.array(val))
print(prediction)