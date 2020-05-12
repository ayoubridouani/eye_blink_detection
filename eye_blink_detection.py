#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2


# In[2]:


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


# In[3]:


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

path_shape_predictor = "./shape_predictor_68_face_landmarks.dat"
path_video = "./blink_detection.mp4"

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_shape_predictor)

# create a videoCapture object with a video file or a capture device
cap = cv2.VideoCapture(path_video)

# check if we will successfully open the file
if not cap.isOpened():
    print("Error opening the file.")
    assert(False)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # Define the codec and create VideoWriter object.The output is stored in 'output.mp4' file.
# out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
# read until the end of the video frame by frame
while cap.isOpened():
    # cap.read (): decodes and returns the next video frame
    # variable ret: will get the return value by retrieving the camera frame, true or false (via "cap")
    # variable frame: will get the next image of the video (via "cap")
    ret, frame = cap.read()

    if ret:
        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # reset the eye frame counter
                COUNTER = 0

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (frame_width - 120, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # to display the frame
        cv2.imshow("Output", frame)
        
        # Write the frame into the file 'output.avi'
        # out.write(frame)

        # waitKey (0): put the screen in pause because it will wait infinitely that key
        # waitKey (n): will wait for keyPress for only n milliseconds and continue to refresh and read the video frame using cap.read ()
        # ord (character): returns an integer representing the Unicode code point of the given Unicode character.
        if cv2.waitKey(1) == ord('e'):
            break
    else:
        break

# to release software and hardware resources
cap.release()
# out.release()

# to close all windows in imshow ()
cv2.destroyAllWindows()

