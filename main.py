import time
import cv2 as cv
import dlib
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist

# Global Configuration Variables
FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
MINIMUM_EAR = 0.22
MAXIMUM_FRAME_COUNT = 1
eye_counter = 0
EYE_CLOSED_COUNTER = 0
data = [0]
timer = [0]

# Initializations
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")
capture = cv.VideoCapture("video/ignore_video/filmik.mp4")
start_time = time.perf_counter()


def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    return (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)


def shape_to_np(shape, dtype="int"):
    cord = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        cord[i] = (shape.part(i).x, shape.part(i).y)
    return cord


while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        faceLandmarks = predictor(gray, face)
        faceLandmarks = shape_to_np(faceLandmarks)

        leftEye = faceLandmarks[42:48]
        rightEye = faceLandmarks[36:42]

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

        cv.putText(
            frame,
            "EAR: {}".format(round(ear, 2)),
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv.putText(
            frame,
            f"Count: {eye_counter}",
            (10, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        data.append(ear)
        timer.append(time.perf_counter() - start_time)
    cv.imshow("Frame", frame)
    if cv.waitKey(20) & 0xFF == ord("q"):
        data.pop(0)
        timer.pop(0)
        d = scipy.signal.medfilt(data, 3)
        plt.plot(timer, d)
        plt.axhline(MINIMUM_EAR, color="r", linestyle="--")
        plt.xlabel("Time [s]")
        plt.ylabel("EAR")
        plt.show()
        break
