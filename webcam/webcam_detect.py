import cv2
import sys
import numpy as np
sys.path.append("../")
import model.basenet as baseNet
import detection_utilities as du

windowName = 'Webcam Screen'
weights_file = 'baseNet_weight.h5'
FACE_SHAPE = (48, 48)

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
        if face_coordinates is not None frame is not None:
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


def main():
    print("Start main() function.")
    model = buildNet(weights_file)
    capture = getCameraStreaming()
    setDefaultCameraSetting()
    showScreenAndDetectFace(model, capture)


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()
