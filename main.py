import csv
import time

import cv2 as cv
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Global Configuration Variables
FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
MINIMUM_EAR = 0.3
MAXIMUM_FRAME_COUNT = 1
eye_counter = 0
EYE_CLOSED_COUNTER = 0
data = [0]
timer = [0]

# Initializations
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")
capture = cv.VideoCapture("video/1.mp4")
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


capture.set(cv.CAP_PROP_POS_FRAMES, 30*4)

counter = 0
while True:
    isTrue, frame_big = capture.read()
    if not isTrue:
        break
    frame = frame_big[300:1000, 200:900]
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
        connected = np.concatenate((rightEye, leftEye), axis=0)
        for x, y in connected:
            cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

        counter += 1
        cv.putText(
            frame,
            "FPS: {}".format(round(ear, 2)),
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv.putText(
            frame,
            f"Count: {counter}",
            (10, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        data.append(ear)
        timer.append(time.perf_counter() - start_time)
    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

with open('data3.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(data)):
        row = [i, data[i]]
        writer.writerow(row)

