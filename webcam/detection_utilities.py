import cv2
import dlib
from imutils import face_utils

# dlib을 위한 변수
landmarks = 'shape_predictor_68_face_landmarks.dat'
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks)

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
RED_COLOR = (0, 0, 255)


# Face Detection using Haar Cascades
# URL : https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
def getFaceCoordinates(img):
    # cv2.CascadeClassifier 특정 객체를 학습시켜 구분하기위함
    # 여기서는 사람 얼굴 검출하기 위함
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    # 회색으로 색 변환.
    imgToGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # frmae_img를 histogramEqualization 해준다.
    # 설명 URL : https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    src_img = cv2.equalizeHist(imgToGray)

    # CascadeClassifier의 detectMultiScale 함수에 grayscale 이미지를 입력하여 얼굴을 검출합니다.
    # 얼굴이 검출되면 위치를 리스트로 리턴합니다.
    # 이 위치는 (x, y, w, h)와 같은 튜플이며 (x, y)는 검출된 얼굴의 좌상단 위치
    # w, h는 가로, 세로 크기입니다.
    coordinates = cascade.detectMultiScale(
        image=src_img,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(48, 48))
    return coordinates


def dlib_face_coordinates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detector(gray, 0)


def drawFace(frame, face_coordinates):
    # case : dlib
    if len(face_coordinates) > 0:
        for face in face_coordinates:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), RED_COLOR, thickness=1)
    # case : haar cascade
    # for (x, y, w, h) in face_coordinates:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), RED_COLOR, thickness=2)


def crop_face(frame, face_coordinates):
    # case : dlib
    cropped_img = frame
    (x, y, w, h) = face_utils.rect_to_bb(face_coordinates[0])
    cropped_img = frame[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    # cv2.imwrite('./0.png', cropped_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

    # case : haar cascade
    # cropped_img = frame
    # for (x, y, w, h) in face_coordinates:
    #     cropped_img = frame[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    return cropped_img


def preprocess(img, face_coordinates, face_shape=(48, 48)):
    face = crop_face(img, face_coordinates)
    face_resize = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./123.png', face_gray, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
    return face_gray


def checkFaceCoordinate(face_coordinates, width, height):
    for (x, y, w, h) in face_coordinates:
        x1 = int(width * 0.25)
        x2 = int(width * 0.5)
        y1 = int(height * 0.25)
        y2 = int(height * 0.5)
        if x in range(x1, x2) and y in range(y1, y2):
            return True
        else:
            return False
