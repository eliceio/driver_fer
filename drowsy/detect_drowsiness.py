# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import time


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


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


##눈 크기를 측정하기 위해 처음 시작을 알리는 webcam 출력값
def start(cv2, repeat):
    if repeat >= 1 and repeat <= 10:
        cv2.putText(frame, "Look at the camera for five seconds.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)
    elif repeat >= 11 and repeat <= 20:
        cv2.putText(frame, "3", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)
    elif repeat >= 21 and repeat <= 30:
        cv2.putText(frame, "2", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)
    elif repeat >= 31 and repeat <= 40:
        cv2.putText(frame, "1", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)


def warning(ALARM_ON):
    # if the alarm is not on, turn it on
    if not ALARM_ON:
        ALARM_ON = True

        if args["alarm"] != "":
            t = Thread(target=sound_alarm,
                       args=(args["alarm"],))
            t.deamon = True
            t.start()

    # draw an alarm on the frame
    cv2.putText(frame, "Drowsy Warning!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return ALARM_ON

##5초동안 사용자 눈 크기 계산
def eye_size_cal(repeat, user_eye, ear, Sleeping_eye):
    user_eye += ear
    print("ear:" + str(ear))
    print("user : " + str(user_eye))
    cv2.putText(frame, "eye size: {:.2f}".format(user_eye / (repeat - 40)), (280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if repeat == 45:
        print("사용자 눈 크기 " + str(user_eye / 5))
        user_eye = (user_eye / 5)
        Sleeping_eye = (user_eye * 0.75)
        print("자는 눈 계산" + str(Sleeping_eye))
    return user_eye, Sleeping_eye


def current_eye_size():
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    return ear

# 인자값, default로 설정 완료
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", default="alarm.WAV",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

##연속적으로 눈을 감은 횟수
EYE_AR_CONSEC_FRAMES = 40
consecutive_eyes_closed = 1

## 평균 눈 깜빡임 횟수
Blink_avg_number = 0

# x초 마다 깜빡임 수를 구하기
x_second = 10
x_repeat = 0

# 전체 눈 깜빡임 횟수
TOTAL = 0
##깜빡인 시간차를 위한 변수
Eye_blinking_time = 0
#눈 뜬 여부
eye_open = True
#연속적으로 눈 깜빡이는 주기가 느린 횟수
count_drowsy_detection=0 #횟수 저장
consecutive_drowsy = 4    #기준이 되는 횟수

##눈 인식 되지 않는 시간##
Eye_Notrecognition_time = 0

##사용자의 눈 크기
user_eye = 0
repeat = 0
Sleeping_eye=0

##졸음이 깰때까지 webcam에 문구를 띄우기 위해 추가 된 변수
Slow_blinking=False

COUNTER = 0
ALARM_ON = False

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
start_time = 0
end_time=0

# loop over frames from the video stream
while True:
    repeat += 1
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.rectangle(frame, (130, 80), (330, 280), (255, 0, 0), 3)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    ##눈 인식이 20초 이상 되지 않을 경우 알림 울리기
    if not rects:
        Eye_Notrecognition_time += 1
    if Eye_Notrecognition_time >= 20:
        ALARM_ON = warning(ALARM_ON)
    if repeat <= 40:
        start(cv2, repeat)


    # loop over the face detections
    for rect in rects:
        Eye_Notrecognition_time = 0
        ear = current_eye_size()



        #사용자의 평균 눈 크기 구하기
        if repeat >= 41 and repeat <=45:
            user_eye, Sleeping_eye = eye_size_cal(repeat, user_eye, ear,Sleeping_eye)


        elif repeat >= 46 and repeat<=55:
            cv2.putText(frame, "Start!", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 0, 0), 2)

        elif repeat>=56:
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < Sleeping_eye:
                if eye_open :
                    print("눈감음")
                    start_time=time.time()
                    eye_open = False

                COUNTER += 1
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    ALARM_ON = warning(ALARM_ON)

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
                        if not Slow_blinking:
                            ALARM_ON = warning(ALARM_ON)
                            Slow_blinking = True

                    if Slow_blinking:
                        print("slow_blink")
                        Slow_blinking = False

                    # print(end_time-start_time)
                    eye_open = True
                if COUNTER >= consecutive_eyes_closed:
                    TOTAL += 1
                COUNTER = 0
                ALARM_ON = False


            # draw the computed eye aspect ratio on the frame to help
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 눈 깜빡인 횟수 화면 출력
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()