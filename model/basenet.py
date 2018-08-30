import numpy as np
import pandas as pd
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

weights_file = 'baseNet_weight.h5'


def load_data():
    print("Start load_data")
    fer = pd.read_csv('./fer2013.csv')
    fer_train = fer[fer.Usage == 'Training']
    fer_test = fer[fer.Usage.str.contains('Test', na=False)]

    x_train = np.array([list(map(int, x.split())) for x in fer_train['pixels'].values])
    y_train = np.array(fer_train['emotion'].values)

    x_test = np.array([list(map(int, x.split())) for x in fer_test['pixels'].values])
    y_test = np.array(fer_test['emotion'].values)

    x_train = normalize_x(x_train)
    x_test = normalize_x(x_test)
    y_train = np_utils.to_categorical(y_train, 7)
    y_test = np_utils.to_categorical(y_test, 7)

    return x_train, x_test, y_train, y_test


def normalize_x(data):
    faces = []

    for face in data:
        face = face.reshape(48, 48) / 255.0
        face = cv2.resize(face, (48, 48))
        faces.append(face)

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    return faces


class BaseNet:
    def __init__(self, weights_file=None):
        self.model = self.buildNet(weights_file)

    def buildNet(self, weights_file=None):
        print("Start buildNet.")
        model = Sequential()
        model.add(Conv2D(64, (5, 5), padding='valid', input_shape=(48, 48, 1), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), padding='valid'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Conv2D(128, (4, 4), padding='valid'))
        model.add(Flatten())
        model.add(Dense(3072))
        model.add(Dense(7, activation='softmax'))

        self.model = model

        print("Create model successfully!")
        if weights_file:
            model.load_weights(weights_file)

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                      metrics=['accuracy'])
        return model

    def train(self, x_train, y_train):
        print("Start training")
        self.model.fit(x_train, y_train,
                       batch_size=64,
                       epochs=100,
                       verbose=1)
        print("Save weights.")
        self.model.save(weights_file)

    def evaluate(self, x_test, y_test):
        print("evaluate test.")
        scores = self.model.evaluate(x_test, y_test, batch_size=64)
        print("Loss:", scores[0])
        print("Accuracy:", scores[1])
        return scores

    def predict(self, x):
        return self.model.predict(x)

    def plot(self):
        plot_model(self.model, to_file='BaseNet.png', show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    print("===============================================================")
    print("loading datasets")
    x_train, x_test, y_train, y_test = load_data()
    print("normalize_x for x_train")
    x_train = normalize_x(x_train)
    print("normalize_x for x_test")
    x_test = normalize_x(x_test)
    print("baseNet\n")
    basenet = BaseNet(weights_file)
    basenet.train(x_train, y_train)
    basenet.evaluate(x_test, y_test)
