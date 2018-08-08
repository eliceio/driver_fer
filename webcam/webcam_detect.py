import cv2
import sys
import detection_utilities as du

windowName = 'Webcam Screen'


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


def showScreenAndDetectFace(capture):
    while True:
        ret, frame = capture.read()
        face_coordinates = du.getFaceCoordinates(frame)
        refreshScreen(frame, face_coordinates)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


def refreshScreen(frame, face_coordinates):
    cropped = frame
    if face_coordinates is not None:
        cropped = du.crop_face(frame, face_coordinates)
        cv2.imshow('crop', cropped)
        du.drawFace(frame, face_coordinates)
    cv2.imshow(windowName, frame)


def main():
    print("Start main() function.")

    capture = getCameraStreaming()
    setDefaultCameraSetting()
    showScreenAndDetectFace(capture)


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()
