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
FACE_SHAPE = (48, 48)

basenet_weight_path = 'baseNet_weight.h5'
# 이런 식으로 나중에 변경.
vgg16_weight_path = 'vgg16_weight.h5'
resnet_weight_path = 'resnet_weight.h5'

emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
isContinue = True
isArea = False
camera_width = 0
camera_height = 0


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


def showScreenAndDetectFace(model, capture):
    global isContinue, isArea
    while True:
        ret, frame = capture.read()
        # case : dlib
        face_coordinates = du.dlib_face_coordinates(frame)
        bounding_box = du.checkFaceCoordinate(face_coordinates)
        # case : haar cascade
        # face_coordinates = du.getFaceCoordinates(frame)

        # 얼굴을 detection 한 경우.
        # case : dlib / if 조건만 다름.
        if len(bounding_box) > 0 and isContinue:
            face = du.preprocess(frame, bounding_box, FACE_SHAPE)
            input_img = np.expand_dims(face, axis=0)
            input_img = np.expand_dims(input_img, axis=-1)
            result = model.predict(input_img)[0]
            index = np.argmax(result)
            print("Emotion : ", emotion[index])
            cv2.putText(frame, emotion[index], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # case : haar cascade / if 문 내용 위와 동일.
        # if len(face_coordinates) is not 0 and du.checkFaceCoordinate(face_coordinates, camera_width, camera_height) and isContinue:

        refreshScreen(frame, bounding_box)
        key = cv2.waitKey(20)
        if key == ord('s'):
            isContinue = not isContinue
        elif key == ord('c'):
            isArea = not isArea
        elif key == ord('q'):
            break


def refreshScreen(frame, bounding_box):
    if isArea:
        du.check_detect_area(frame)
    du.drawFace(frame, bounding_box)
    # case : haar cascade
    # global camera_width, camera_height
    # if du.checkFaceCoordinate(face_coordinates, camera_width, camera_height):
    #     du.drawFace(frame, face_coordinates)
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
