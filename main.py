import time
import cv2 as cv
import dlib
import scipy
from imutils import face_utils
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist

#Global Configuration Variables
FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  # path to dlib's pre-trained facial landmark predictor
MINIMUM_EAR = 0.22   # Minimum EAR for both the eyes to mark the eyes as open
MAXIMUM_FRAME_COUNT = 1  # Maximum number of consecutive frames in which EAR can remain less than MINIMUM_EAR, otherwise alert drowsiness
eye_counter = 0

#Initializations
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks_GTX.dat')
capture = cv.VideoCapture('filmik.mp4')

# Finding landmark id for left and right eyes
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


EYE_CLOSED_COUNTER = 0
data = [0]
timer = [0]
start_time = time.perf_counter()
while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        faceLandmarks = predictor(gray, face)
        faceLandmarks = face_utils.shape_to_np(faceLandmarks)

        leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
        rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)

        cv.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 2)
        cv.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 2)

        if ear < MINIMUM_EAR:
            EYE_CLOSED_COUNTER += 1
        else:
            if EYE_CLOSED_COUNTER > 0:
                eye_counter += 1
            EYE_CLOSED_COUNTER = 0

        cv.putText(frame, "EAR: {}".format(round(ear, 2)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
        cv.putText(frame, f"Count: {eye_counter}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        data.append(ear)
        timer.append(time.perf_counter() - start_time)
        # dateTimeObj = datetime.now()
        # timestampStr = dateTimeObj.strftime("(%H:%M:%S.%f)")
        # time.append(timestampStr)
    cv.imshow("Frame", frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        data.pop(0)
        timer.pop(0)
        d = scipy.signal.medfilt(data, 3)
        plt.plot(timer, d)
        step = len(timer) // 3  # liczba danych dzielona przez liczbę etykiet
        # labels = [0, step, 2 * step, 3 * step]
        # plt.xticks(labels)
        plt.show()
        break