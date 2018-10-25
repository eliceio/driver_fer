import argparse
import cv2
import sys
import numpy as np

sys.path.append("../")
import model.basenet as baseNet
import detection_utilities as du
# from detection_utilities import end_time, start_time, user_eye

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

import os
import glob
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle
import imutils
import requests

from imutils.video import VideoStream
from scipy.misc import imsave

import matplotlib.animation as animation
from matplotlib import style
from io import StringIO

import json

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

class_emotion = ['angry', 'happy', 'neutral']
mpl.style.use('fivethirtyeight')
# plt.style.context('fivethirtyeight'):

parser = argparse.ArgumentParser(description="운전자 졸음, 난폭 운전 예방 시스템")
parser.add_argument('model', type=str, default='ak',
                    choices=['ak', 'ak_weak', 'ak8', 'mobile', 'basenet', 'vgg16', 'resnet', 'ensemble'],
                    help="운전자 감정 예측을 위한 모델을 선택")
args = parser.parse_args()
model_name = args.model

windowName = 'Webcam Screen'
FACE_SHAPE = (48, 48)

#### model list

# model_list = os.listdir('../model/models')
model_list = glob.glob('../model/models/*.h5')
print(model_list)  # model list preparation

ak_path = '../model/models/ak31_32.h5'  # jj_add / model path
ak_weak_path = '../model/models/ak_weak_weak.h5'
ak8_path = '../model/models/ak8.h5'
basenet_weight_path = '../model/models/base_3.h5'
# 이런 식으로 나중에 변경.
vgg16_weight_path = 'vgg16_weight.h5'
resnet_weight_path = 'resnet_weight.h5'

mobile_path = '../model/models/test_mobile_model.h5'  # jj_add / model path
mobile_weight_path = '../model/models/test_mobile_weight.h5'  # jj_add / model path

emotion_7_class = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

isContinue = True
isArea = True
isLandmark = False
isServing = False
isGragh = False
capture_cnt = 0
camera_width = 0
camera_height = 0
input_img = None
rect = None
bounding_box = None

# model_url = 'http://143.248.92.116:8002/model/predict_images/images'
model_url = 'http://143.248.92.116:8002/model/serving_default/input_image'

import matplotlib as mpl
from scipy import signal

class_emotion = ['angry', 'happy', 'neutral']
# class_drowsy = ['eye blink speed', 'eye size ratio']
class_drowsy = ['eye size']
result = []


def plot_hist(emotion_hist, class_hist):
    emotion_hist = np.array(emotion_hist) * 100  # % 로 표현

    n = len(class_hist)
    t = emotion_hist.reshape((-1, n))

    t = signal.resample(t, int(len(t) / 5))
    x = np.arange(t.shape[0])

    fig, ax = plt.subplots()

    for i in range(n):
        ax.plot(x, t[:, i], 'o-', label=class_hist[i])
    if n == 1:
        name = 'drowsy'
        ax.set(title='Drowsy history', ylabel='eye size', xlabel='Time')
    elif n == 3:
        ax.set(title='Emotion history tracking', ylabel='Prediction (%)', xlabel='Time')
        # ax = plt.axes(xlim=(0,100), ylim=(-10, 150))
        name = 'emotion'

    fig.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('../data/plot_' + name + '_hist')
    fig.show()
    plt.pause(2)


def getCameraStreaming():
    capture = cv2.VideoCapture(0)
    global camera_width, camera_height
    camera_width = capture.get(3)
    camera_height = capture.get(4)
    du.set_default_min_max_area(camera_width, camera_height)
    if not capture:
        print("Failed to capture video streaming")
        sys.exit()
    print("Successed to capture video streaming")
    return capture


def setDefaultCameraSetting():
    cv2.startWindowThread()
    cv2.namedWindow(winname=windowName)
    cv2.setWindowProperty(winname=windowName, prop_id=cv2.WINDOW_FULLSCREEN, prop_value=cv2.WINDOW_FULLSCREEN)


def showScreenAndDetectFace(model, capture, emotion, color_ch=1):  # jj_add / for different emotion class models
    global isContinue, isServing, isGragh, isArea, isLandmark, input_img, rect, bounding_box, result

    img_counter = 1  # jj_add / for counting images that are saved (option)
    emotion_hist = []
    drowsy_hist = []

    #### For live plot
    # plt.show()
    line1 = []
    line2 = []
    line3 = []
    list_line = [line1, line2, line3]

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 100), ylim=(-10, 150))
    ax.set(title='Emotion history tracking', ylabel='Prediction (%)', xlabel='Time')
    # axes = plt.gca()

    for i in range(3):
        list_line[i], = ax.plot([], [], 'o-', linewidth=3, label=class_emotion[i])

    ax.legend(loc='upper right')

    ###############

    while True:
        input_img, rect, bounding_box = None, None, None
        ret, frame = capture.read()
        face_coordinates = du.dlib_face_coordinates(frame)
        # if isContinue:
        eye_speed, ear_out, user_eye = detect_area_driver(frame, face_coordinates, color_ch)

        if not isContinue:
            if img_counter in range(1, 31):
                cv2.putText(frame, "Image capture mode (Space bar)",
                            (int(camera_width * 0.2), int(camera_height * 0.15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "finish image capture", (int(camera_width * 0.31), int(camera_height * 0.15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if img_counter in range(1, 11):
                cv2.putText(frame, "Angry Img ({}/{})".format(img_counter, 10),
                            (int(camera_width * 0.35), int(camera_height * 0.85)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif img_counter in range(11, 21):
                cv2.putText(frame, "Happy Img ({}/{})".format(img_counter - 10, 10),
                            (int(camera_width * 0.35), int(camera_height * 0.85)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif img_counter in range(21, 31):
                cv2.putText(frame, "Neutral Img ({}/{})".format(img_counter - 20, 10),
                            (int(camera_width * 0.35), int(camera_height * 0.85)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if isContinue and input_img is not None:

            if du.repeat >= 56:
                if isServing:  # serving 이 무조건 된다는 조건임...
                    http_post_array(input_img)  # for serving
                    cv2.putText(frame, "Serving mode", (int(camera_width * 0.05), int(camera_height * 0.95)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    result = model.predict(input_img)[0]
                    cv2.putText(frame, "Local mode", (int(camera_width * 0.05), int(camera_height * 0.95)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                index = int(np.argmax(result))

                for i in range(len(emotion)):
                    # print("Emotion :{} / {} % ".format(emotion[i], round(result[i]*100, 2)))
                    cv2.putText(frame, "{}: {}% ".format(emotion[i], round(result[i] * 100, 2)), (5, 20 + (i * 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, "Driver emotion: {}".format(emotion[index]), (5, 20 + (20 * len(emotion))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                du.add_driver_emotion(index)
                du.check_driver_emotion(frame)

                if isGragh:
                    cv2.putText(frame, "Graph ON", (int(camera_width * 0.7), int(camera_height * 0.95)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 50, 200), 2)
                    # History saving
                    emotion_hist.append(result)  # to track emotion history
                    eye_size = ear_out  # to prevent divide by zero
                    drowsy_hist.append([eye_size])  # eye history

                    ## live plot
                    n_emotion = len(emotion_hist)
                    # print(str(n_emotion)+'\n')

                    if n_emotion % 4 == 0:
                        emotion_data = np.array(emotion_hist)
                        # x_n = np.shape(emotion_data)[0]
                        xdata = np.array(range(n_emotion))

                        # 3종류 감정 각각 누적 데이터 plot
                        for i in range(3):
                            list_line[i].set_data(xdata, emotion_data[:, i] * 100)

                        ax.set_xlim(0, n_emotion)
                        fig.tight_layout()

                        plt.draw()
                        plt.pause(0.1)
                        # time.sleep(0.1)

                        # add this if you don't want l the window to disappear at the end
                        # plt.show()
                else:
                    cv2.putText(frame, "Graph OFF", (int(camera_width * 0.7), int(camera_height * 0.95)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        refreshScreen(frame)
        key = cv2.waitKey(20)
        if key == ord('s'):
            img_counter = 1
            isContinue = not isContinue
        elif key == ord('m'):  # change serving mode, local model mode
            isServing = not isServing
        elif key == ord('g'):
            isGragh = not isGragh
        elif key == ord('c'):
            isArea = not isArea
        elif key == ord('l'):
            isLandmark = not isLandmark
        elif key == ord('o'):
            du.expend_detect_area()
        elif key == ord('p'):
            du.reduce_detect_area()
        elif key == ord('r'):
            du.reset_global_value()
        elif key == ord('q'):
            time_now = datetime.now().strftime('%Y%m%d_%H%M%S')  # save and plot emotion history
            np.save('../data/' + time_now + 'hist_emotion.npy', emotion_hist)
            plot_hist(emotion_hist, class_emotion)
            np.save('../data/' + time_now + 'hist_drowsy.npy', drowsy_hist)
            plot_hist(drowsy_hist, class_drowsy)

            break

        elif key % 256 == 32:  # jj_add / press space bar to save cropped gray image
            img_counter = user_img_capture(img_counter)


def user_img_capture(img_counter):
    try:
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        path1= '../data/_transfer/angry/'
        path2= '../data/_transfer/happy/'
        path3='../data/_transfer/neutral/'
        
        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)
        if not os.path.exists(path3):
            os.makedirs(path3)
            
        if img_counter in range(1, 11):
            img_name = '../data/_transfer/angry/' + time_now + "_cropped_gray_{}.png".format(img_counter)
        elif img_counter in range(11, 21):
            img_name = '../data/_transfer/happy/' + time_now + "_cropped_gray_{}.png".format(img_counter)
        elif img_counter in range(21, 31):
            img_name = '../data/_transfer/neutral/' + time_now + "_cropped_gray_{}.png".format(img_counter)
        cv2.imwrite(img_name,
                    np.squeeze(input_img * 255.))  # to recover normalized img to save as gray scale image
        print("{} written!".format(img_name))
        img_counter += 1
    except:
        print('Image can not be saved!')
    return img_counter


def detect_area_driver(frame, face_coordinates, color_ch=1):
    global input_img, rect, bounding_box
    rect, bounding_box = du.checkFaceCoordinate(face_coordinates, isArea)
    # 얼굴을 detection 한 경우.
    if bounding_box is not None:
        # du.drowsy_detection(frame, rect)
        face = du.preprocess(frame, bounding_box, FACE_SHAPE)
        if face is not None:
            input_img = np.expand_dims(face, axis=0)
            # input_img = np.expand_dims(input_img, axis=-1)
            input_img = np.stack((input_img,) * color_ch, -1)
            # print(np.mean(input_img))

    if isContinue:
        eye_speed, ear_out, user_eye = du.drowsy_detection(frame, rect)
        return eye_speed, ear_out, user_eye
    return None, None, None


def http_post_array(data):
    global result
    data = np.squeeze(data, axis=0)
    data = np.squeeze(data, axis=2)
    s = StringIO()
    np.savetxt(s, data, delimiter=',')
    csv_str = s.getvalue()

    rsp = requests.post(model_url, data=csv_str,
                        headers={'Content-Type': 'text/csv'})
    print(rsp.status_code, rsp.reason)
    print(rsp.headers)
    d = json.loads(rsp.text)
    result = d['outputs']['activation_4_2/Softmax:0']['floatVal']


def http_post():
    # model_url = 'http://143.248.92.116:8002/model/serving_default/input_image'

    with open('./face.jpg', 'rb') as jpeg_file:
        jpeg_bytes = jpeg_file.read()
        rsp = requests.post(model_url, data=jpeg_bytes,
                            headers={'Content-Type': 'image/jpeg'})
        print(rsp.status_code, rsp.reason)
        print(rsp.headers)
        print(rsp.text)


def refreshScreen(frame):
    if isArea:
        du.check_detect_area(frame)
    if isLandmark:
        du.draw_landmark(frame, rect)
    # if bounding_box is not None:
    du.drawFace(frame, bounding_box)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = clahe.apply(frame)
    cv2.imshow(windowName, frame)


def buildNet(weights_file):
    return baseNet.BaseNet(weights_file)


def chooseWeight(model_name):
    if model_name == 'basenet':
        emotion = ['Angry', 'Happy', 'Neutral']
        return basenet_weight_path, emotion
    elif model_name == 'vgg16':
        return vgg16_weight_path
    elif model_name == 'resnet':
        return resnet_weight_path
    elif model_name == 'ak':
        emotion = ['Angry', 'Happy',
                   'Neutral']  ## jj_add /  3 emotion classes for ak net. return path and emotion classes
        return ak_path, emotion
    elif model_name == 'ak_weak':
        emotion = ['Angry', 'Happy',
                   'Neutral']  ## jj_add /  3 emotion classes for ak net. return path and emotion classes
        return ak_weak_path, emotion
    elif model_name == 'ak8':
        emotion = ['Angry', 'Happy',
                   'Neutral']  ## jj_add /  3 emotion classes for ak net. return path and emotion classes
        return ak8_path, emotion

    elif model_name == 'mobile':
        emotion = ['Angry', 'Happy', 'Neutral']
        return [mobile_path, mobile_weight_path], emotion


def main():
    print("Start main() function.")
    model_weight_path, emotion = chooseWeight(model_name)
    color_ch = 1  # default for gray

    if model_name == 'mobile':  # mobilenet needs custom object
        with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
            model = load_model(model_weight_path[0])
        model.load_weights(model_weight_path[1])
        color_ch = 3
    else:
        model = load_model(model_weight_path)
        # model = buildNet(model_weight_path)

    capture = getCameraStreaming()
    setDefaultCameraSetting()
    showScreenAndDetectFace(model, capture, emotion, color_ch)  # jj_add / for different emotion class models


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()
