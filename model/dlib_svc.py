#Import required modules
import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.externals import joblib
import time
from functools import wraps

def time_fn(fn):
    @wraps(fn)
    def time_measure(*args, **kwargs):
        t1 = time.time()
        function_result = fn(*args, **kwargs)
        t2 = time.time()
        print('@time_fn: {} took {} sec.'.format(fn.__name__, t2 - t1))
        return function_result
    return time_measure

emotions = ["angry", "not-angry"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3, max_iter=10000)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []

        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vectorised'] = "error"

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        cnt = 0
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                # print("no face detected on this one")
                cnt += 1
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))
        print(cnt/len(training_data))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                # print("no face detected on this one")
                pass
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels
accur_lin = []

@time_fn
def training(X,Y):
    return clf.fit(npar_train, npar_trainlabs)

for i in range(0,10):
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels).reshape(-1, 1)
    print("training SVM linear %s" %i) #train SVM
    clf = training(npar_train, npar_trainlabs)
    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    npar_trainlabs = np.array(prediction_labels).reshape(-1, 1)
    pred_lin = clf.score(npar_pred, npar_trainlabs)
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list
joblib.dump(clf, 'models/svc.pkl')
print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs





#Set up some required objects
# video_capture = cv2.VideoCapture(0) #Webcam object
# filename = 'IMG_5573.JPG'
# detector = dlib.get_frontal_face_detector() #Face detector
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file


# def get_from_cam():
#     while True:
#         ret, frame = video_capture.read()
#         frame = cv2.imread(filename)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         clahe_image = clahe.apply(gray)
#         detections = detector(clahe_image, 1) #Detect the faces in the image
#         for k,d in enumerate(detections): #For each detected face
#             shape = predictor(clahe_image, d) #Get coordinates
#             for i in range(1,68): #There are 68 landmark points on each face
#                 cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
#         cv2.imshow("image", frame) #Display the frame
#         if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
#             break
#
# def get_landmarks(image):
#     frame = cv2.imread(image)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     clahe_image = clahe.apply(gray)
#     detections = detector(clahe_image, 1)
#     for k, d in enumerate(detections):  # For all detected face instances individually
#         shape = predictor(frame, d)  # Draw Facial Landmarks with the predictor class
#         xlist = []
#         ylist = []
#         for i in range(1, 68):  # Store X and Y coordinates in two lists
#             xlist.append(float(shape.part(i).x))
#             ylist.append(float(shape.part(i).y))
#         xmean = np.mean(xlist)  # Find both coordinates of centre of gravity
#         ymean = np.mean(ylist)
#         xcentral = [(x - xmean) for x in xlist]  # Calculate distance centre <-> other points in both axes
#         ycentral = [(y - ymean) for y in ylist]
#         landmarks_vectorised = []
#         for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
#             landmarks_vectorised.append(w)
#             landmarks_vectorised.append(z)
#             meannp = np.asarray((ymean, xmean))
#             coornp = np.asarray((z, w))
#             dist = np.linalg.norm(coornp - meannp)
#             landmarks_vectorised.append(dist)
#             landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
#         data['landmarks_vectorised'] = landmarks_vectorised
#     if len(detections) < 1:
#         data['landmarks_vestorised'] = "error"

