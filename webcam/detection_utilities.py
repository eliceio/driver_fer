import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import time
import sys

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
angry_check=0
# dlib을 위한 변수
landmarks = '../src/shape_predictor_68_face_landmarks.dat'  # jj_modify for relative path to the dat
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks)

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
RED_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)

cam_width, cam_height = 0, 0
expand_width, expand_height = 0, 0
reduce_width, reduce_height = 0, 0
min_x, max_x, min_y, max_y = 0, 0, 0, 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

emotion_list = []


def dlib_face_coordinates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    return detector(gray, 0)


def drawFace(frame, face_coordinates):
    if face_coordinates is not None:
        (x, y, w, h) = face_coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), RED_COLOR, thickness=1)


def crop_face(frame, face_coordinates):
    cropped_img = frame
    (x, y, w, h) = face_coordinates
    if check_resize_area(face_coordinates):
        cropped_img = frame[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        # cv2.imwrite('./0.png', cropped_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        return cropped_img
    else:
        return None


def check_resize_area(face_coordinates):
    (x, y, w, h) = face_coordinates
    if x - int(w / 4) > 0 and y - int(h / 4) > 0:
        return True
    return False


def preprocess(img, face_coordinates, face_shape=(48, 48)):
    face = crop_face(img, face_coordinates)
    if face is not None:
        face_resize = cv2.resize(face, face_shape)
        face_gray = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)
        face_gray = clahe.apply(face_gray) / 255.
        cv2.imwrite('./123.png', face_gray, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        return face_gray
    else:
        return None


def set_default_min_max_area(width, height):
    global cam_width, cam_height, min_x, max_x, min_y, max_y, \
        expand_width, expand_height, reduce_width, reduce_height
    cam_width, cam_height = width, height
    min_x = int(width * 0.2)
    max_x = int(width * 0.8)
    min_y = int(height * 0.2)
    max_y = int(height * 0.8)
    expand_width = int(cam_width * 0.05)
    expand_height = int(cam_height * 0.05)
    reduce_width = int(cam_width * 0.05)
    reduce_height = int(cam_height * 0.05)


def expend_detect_area():
    global min_x, max_x, min_y, max_y
    min_x -= expand_width
    min_y -= expand_height
    max_x += expand_width
    max_y += expand_height


def reduce_detect_area():
    global min_x, max_x, min_y, max_y
    min_x += reduce_width
    min_y += reduce_height
    max_x -= reduce_width
    max_y -= reduce_height


def check_detect_area(frame):
    cv2.line(frame, (min_x, min_y), (min_x, max_y), WHITE_COLOR, 2)
    cv2.line(frame, (min_x, max_y), (max_x, max_y), WHITE_COLOR, 2)
    cv2.line(frame, (min_x, min_y), (max_x, min_y), WHITE_COLOR, 2)
    cv2.line(frame, (max_x, min_y), (max_x, max_y), WHITE_COLOR, 2)


def draw_landmark(frame, rect):
    if rect is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


def checkFaceCoordinate(face_coordinates, in_area=True):
    if len(face_coordinates) > 0:
        if in_area:
            for face in face_coordinates:
                (x, y, w, h) = face_utils.rect_to_bb(face)
                if x in range(min_x, max_x) and y in range(min_y, max_y) \
                        and x + w in range(min_x, max_x) and y + h in range(min_y, max_y):
                    return face, (x, y, w, h)
        else:
            face = face_coordinates[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            return face, (x, y, w, h)
    return None, None


def add_driver_emotion(idx_emotion):
    global emotion_list
    # emotion_list에 감정 저장.
    if len(emotion_list) in range(0, 10):
        emotion_list.append(idx_emotion)
    else:
        del emotion_list[0]
        emotion_list.append(idx_emotion)


def check_driver_emotion(frame):
    if len(emotion_list) == 10:
        count = emotion_list.count(0)
        if count > 5:
            sentence = "Angry detection!"
            angry_check = 1
            if ALARM_count == 0:
                warning(frame,sentence)
            else:
                ALARM_end = time.time()
                temp = (ALARM_end - ALARM_start)
                if temp > float(33.4):
                    warning(frame,sentence)
            angry_check=0


# 이 아래는 drowsy code.
# 연속적으로 눈을 감은 횟수
EYE_AR_CONSEC_FRAMES = 15
consecutive_eyes_closed = 1

# 전체 눈 깜빡임 횟수
TOTAL = 0
# 눈 뜬 여부
eye_open = True
# 연속적으로 눈 깜빡이는 주기가 느린 횟수
count_drowsy_detection = 0  # 횟수 저장
consecutive_drowsy = 4  # 기준이 되는 횟수

# 눈 인식 되지 않는 시간##
eye_not_recognition_time = 0

# 사용자의 눈 크기
user_eye = 0
repeat = 0
Sleeping_eye = 0

# 졸음이 깰때까지 webcam에 문구를 띄우기 위해 추가 된 변수
COUNTER = 0
start_time = 0
end_time = 0
# 알림음 시간
ALARM_start = 0
ALARM_end = 0
ALARM_count = 0


def reset_global_value():
    global EYE_AR_CONSEC_FRAMES, consecutive_eyes_closed, TOTAL, eye_open, \
        count_drowsy_detection, consecutive_drowsy, eye_not_recognition_time, \
        user_eye, repeat, Sleeping_eye, COUNTER, start_time, end_time, ALARM_start, ALARM_end, ALARM_count
    EYE_AR_CONSEC_FRAMES = 15
    consecutive_eyes_closed = 1

    TOTAL = 0
    eye_open = True
    count_drowsy_detection = 0  # 횟수 저장
    consecutive_drowsy = 4  # 기준이 되는 횟수

    eye_not_recognition_time = 0

    user_eye = 0
    repeat = 0
    Sleeping_eye = 0

    COUNTER = 0
    start_time = 0
    end_time = 0
    ALARM_start = 0
    ALARM_end = 0
    ALARM_count = 0


# 5초동안 사용자 눈 크기 계산
def eye_size_cal(ear, frame):
    global repeat, user_eye, Sleeping_eye
    user_eye += ear
    # print("ear:" + str(ear))
    # print("user : " + str(user_eye))

    cv2.putText(frame, "eye size: {:.2f}".format(user_eye / (repeat - 40)), (int(cam_width * 0.7), 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if repeat == 45:
        print("사용자 눈 크기 " + str(user_eye / 5))
        user_eye = (user_eye / 5)
        Sleeping_eye = (user_eye * 0.75)
        print("자는 눈 계산" + str(Sleeping_eye))
    return user_eye, Sleeping_eye


def draw_eye(frame, rect):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    ear = (left_ear + right_ear) / 2.0

    left_eye_hull = cv2.convexHull(left_eye)
    right_eye_hull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
    return ear


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# 눈 크기를 측정하기 위해 처음 시작을 알리는 webcam 출력값
def start(frame):
    global repeat
    if repeat in range(1, 11):
        cv2.putText(frame, "Look at the camera for five seconds.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)
    elif repeat in range(11, 21):
        cv2.putText(frame, "3", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)
    elif repeat in range(21, 31):
        cv2.putText(frame, "2", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)
    elif repeat in range(31, 41):
        cv2.putText(frame, "1", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


def warning(frame,sentence):
    global ALARM_start, ALARM_count
    if sentence == "Out of frame!":
        cv2.putText(frame, str(sentence), (int(cam_width * 0.4), int(cam_height * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        t = Thread(target=sound_alarm,
                   args=("../src/alarm_focus.WAV",))
    elif sentence == "Drowsy detection!":
        cv2.putText(frame, str(sentence), (int(cam_width * 0.3), int(cam_height * 0.9)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        t = Thread(target=sound_alarm,
                   args=("../src/alarm_drowsy.WAV",))
    elif sentence == "The blink is slow!":
        cv2.putText(frame, str(sentence), (int(cam_width * 0.3), int(cam_height * 0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        t = Thread(target=sound_alarm,
                   args=("../src/alarm_drowsy.WAV",))
    elif sentence == "Angry detection!":
        cv2.putText(frame, str(sentence), (int(cam_width * 0.3), int(cam_height * 0.9)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        t = Thread(target=sound_alarm,
                   args=("../src/Gymnopedies_Satie.WAV",))
    t.deamon = True
    t.start()
    ALARM_start = time.time()
    ALARM_count += 1


def drowsy_detection(frame, face):
    global repeat, eye_not_recognition_time, user_eye, Sleeping_eye, eye_open, COUNTER, end_time, start_time, count_drowsy_detection, TOTAL, ALARM_end, ALARM_start
    repeat += 1

    # 눈 인식이 20초 이상 되지 않을 경우 알림 울리기
    if not face:
        eye_not_recognition_time += 1
    if eye_not_recognition_time >= 20:
        sentence = "Out of frame!"
        if ALARM_count == 0:
            warning(frame,sentence)
        else:
            ALARM_end = time.time()
            temp = (ALARM_end - ALARM_start)
            if temp > float(2.5):
                warning(frame,sentence)

    if repeat <= 40:
        start(frame)

    if face is not None:
        eye_not_recognition_time = 0
        ear = draw_eye(frame, face)

        # 사용자의 평균 눈 크기 구하기
        if repeat in range(41, 46):
            user_eye, Sleeping_eye = eye_size_cal(ear, frame)
        elif repeat in range(46, 56):
            cv2.putText(frame, "Start!", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        elif repeat >= 56:
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < Sleeping_eye:
                if eye_open:
                    print("눈감음")
                    start_time = time.time()
                    eye_open = False

                COUNTER += 1
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    sentence = "Drowsy detection!"

                    if angry_check == 0:
                        if ALARM_count == 0:
                            warning(frame,sentence)
                        else:
                            ALARM_end = time.time()
                            temp = (ALARM_end - ALARM_start)
                            if temp > float(2.5):
                                warning(frame,sentence)


            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                if not eye_open:
                    end_time = time.time()
                    print("눈 뜸 " + str(end_time - start_time))
                    if (end_time - start_time) >= float(0.4):
                        count_drowsy_detection += 1

                    else:
                        count_drowsy_detection = 0
                    if count_drowsy_detection >= consecutive_drowsy:
                        sentence = "The blink is slow!"
                        if ALARM_count == 0:
                            warning(frame,sentence)
                        else:
                            ALARM_end = time.time()
                            temp = (ALARM_end - ALARM_start)
                            if temp > float(2.5):
                                warning(frame,sentence)
                    eye_open = True
                if COUNTER >= consecutive_eyes_closed:
                    TOTAL += 1
                COUNTER = 0

            # draw the computed eye aspect ratio on the frame to help
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (int(cam_width * 0.8), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "Slow blink count : " + str(count_drowsy_detection), (int(cam_width * 0.65), 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 눈 깜빡인 횟수 화면 출력
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (int(cam_width * 0.45), 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)