import argparse
import cv2
import sys
import numpy as np

sys.path.append("../")
import model.basenet as baseNet
import detection_utilities as du

from keras.models import load_model

parser = argparse.ArgumentParser(description="운전자 졸음, 난폭 운전 예방 시스템")
parser.add_argument('model', type=str, default='basenet', choices=['ak','basenet', 'vgg16', 'resnet', 'ensemble'],
                    help="운전자 감정 예측을 위한 모델을 선택")
args = parser.parse_args()
model_name = args.model

windowName = 'Webcam Screen'
FACE_SHAPE = (48, 48)

basenet_weight_path = 'baseNet_weight.h5'
# 이런 식으로 나중에 변경.
vgg16_weight_path = 'vgg16_weight.h5'
resnet_weight_path = 'resnet_weight.h5'
ak_path = '../model/models/ak_3class_transfer.h5'  #jj_add / model path

emotion_7_class = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

isContinue = True
isArea = True
isLandmark = False
camera_width = 0
camera_height = 0
input_img = None
rect = None
bounding_box = None


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



def showScreenAndDetectFace(model, capture, emotion):  #jj_add / for different emotion class models
    global isContinue, isArea, isLandmark, input_img, rect, bounding_box

    img_counter = 0  # jj_add / for counting images that are saved (option)

    while True:
        input_img, rect, bounding_box = None, None, None
        ret, frame = capture.read()
        face_coordinates = du.dlib_face_coordinates(frame)

        if isContinue:
            detect_area_driver(frame, face_coordinates)

        if input_img is not None:
            result = model.predict(input_img)[0]
            index = int(np.argmax(result))
            
            if du.repeat >= 56:
                print("Emotion :{} / {} % ".format(emotion[index], result[index]*100))
                cv2.putText(frame, emotion[index], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        refreshScreen(frame)
        key = cv2.waitKey(20)
        if key == ord('s'):
            isContinue = not isContinue
        elif key == ord('c'):
            isArea = not isArea
        elif key == ord('l'):
            isLandmark = not isLandmark
        elif key == ord('o'):
            du.expend_detect_area()
        elif key == ord('p'):
            du.reduce_detect_area()
        elif key == ord('q'):
            break
        elif key%256 == 32:  # jj_add / press space bar to save cropped gray image
            try:
                img_name = "cropped_gray_{}.png".format(img_counter)

                cv2.imwrite(img_name, np.squeeze(input_img))
                print("{} written!".format(img_name))
                img_counter += 1
            except:
                print('Image can not be saved!')                

def detect_area_driver(frame, face_coordinates):
    global input_img, rect, bounding_box
    rect, bounding_box = du.checkFaceCoordinate(face_coordinates, isArea)
    du.drowsy_detection(frame, rect)

    # 얼굴을 detection 한 경우.
    if bounding_box is not None and isContinue:
        face = du.preprocess(frame, bounding_box, FACE_SHAPE)
        if face is not None:
            input_img = np.expand_dims(face, axis=0)
            input_img = np.expand_dims(input_img, axis=-1)


def refreshScreen(frame):
    if isArea:
        du.check_detect_area(frame)
    if isLandmark:
        du.draw_landmark(frame, rect)
    # if bounding_box is not None:
    du.drawFace(frame, bounding_box)
    cv2.imshow(windowName, frame)


def buildNet(weights_file):
    return baseNet.BaseNet(weights_file)


def chooseWeight(model_name):
    if model_name == 'basenet':
        return basenet_weight_path, emotion_7_class
    elif model_name == 'vgg16':
        return vgg16_weight_path
    elif model_name == 'resnet':
        return resnet_weight_path
    elif model_name=='ak':
        emotion=['Angry','Happy','Neutral']  ## jj_add /  3 emotion classes for ak net. return path and emotion classes
        return ak_path, emotion


def main():
    print("Start main() function.")
    model_weight_path, emotion = chooseWeight(model_name)

    if model_name =='ak':   ## jj_add /  if model name is ak, than just load_model (without compile?)
        model = load_model(model_weight_path)
    else:
        model = buildNet(model_weight_path)

    capture = getCameraStreaming()
    setDefaultCameraSetting()
    showScreenAndDetectFace(model, capture, emotion)  #jj_add / for different emotion class models


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()
