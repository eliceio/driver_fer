import argparse
import cv2
import sys
import numpy as np

sys.path.append("../")
import model.basenet as baseNet
import detection_utilities as du

parser = argparse.ArgumentParser(description="운전자 졸음, 난폭 운전 예방 시스템")
parser.add_argument('model', type=str, default='basenet', choices=['basenet', 'vgg16', 'resnet', 'ensemble'],
                    help="운전자 감정 예측을 위한 모델을 선택")
args = parser.parse_args()
model_name = args.model

windowName = 'Webcam Screen'
weights_file = 'baseNet_weight.h5'
FACE_SHAPE = (48, 48)

basenet_weight_path = 'baseNet_weight.h5'
# 이런 식으로 나중에 변경.
vgg16_weight_path = 'vgg16_weight.h5'
resnet_weight_path = 'resnet_weight.h5'

emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def getCameraStreaming():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming")
        sys.exit()
    print("Successed to capture video streaming")
    return capture


def setDefaultCameraSetting():
    cv2.startWindowThread()
    cv2.namedWindow(winname=windowName)
    cv2.setWindowProperty(winname=windowName, prop_id=cv2.WINDOW_FULLSCREEN, prop_value=cv2.WINDOW_FULLSCREEN)


def showScreenAndDetectFace(model, capture):
    while True:
        ret, frame = capture.read()
        face_coordinates = du.getFaceCoordinates(frame)
        refreshScreen(frame, face_coordinates)
        # 얼굴을 detection 한 경우.
        if face_coordinates is not None and frame is not None:
            face = du.preprocess(frame, face_coordinates, FACE_SHAPE)
            input_img = np.expand_dims(face, axis=0)
            input_img = np.expand_dims(input_img, axis=-1)
            result = model.predict(input_img)[0]
            index = np.argmax(result)
            print("Emotion : ", emotion[index])
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


def refreshScreen(frame, face_coordinates):
    if face_coordinates is not None:
        du.drawFace(frame, face_coordinates)
    cv2.imshow(windowName, frame)


def buildNet(weights_file):
    return baseNet.BaseNet(weights_file)


def chooseWeight(model_name):
    if model_name == 'basenet':
        return basenet_weight_path
    elif model_name == 'vgg16':
        return vgg16_weight_path
    elif model_name == 'resnet':
        return resnet_weight_path


def main():
    print("Start main() function.")
    model_weight_path = chooseWeight(model_name)
    model = buildNet(model_weight_path)
    capture = getCameraStreaming()
    setDefaultCameraSetting()
    showScreenAndDetectFace(model, capture)


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()
