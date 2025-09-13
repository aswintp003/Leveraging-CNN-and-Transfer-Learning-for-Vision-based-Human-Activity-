# USAGE
# python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4
# python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

# import the necessary packages
import numpy as np
import argparse
import imutils
import sys
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True,
	help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="",
	help="optional path to video file")
args = vars(ap.parse_args())

# load the contents of the class labels file, then define the sample
# duration (i.e., # of frames for classification) and sample size
# (i.e., the spatial dimensions of the frame)
CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(args["model"])

# grab a pointer to the input video stream
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

if not vs.isOpened():
    print("[ERROR] Cannot open video source")
    sys.exit(1)

# loop until we explicitly break from it
while True:
    frames = []
    for i in range(0, SAMPLE_DURATION):
        (grabbed, frame) = vs.read()
        if not grabbed:
            print("[INFO] no frame read from stream - exiting")
            vs.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    blob = cv2.dnn.blobFromImages(frames, 1.0,
        (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
        swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]

    quit_flag = False
    for frame in frames:
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2)
        cv2.imshow("Activity Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            quit_flag = True
            break

    if quit_flag:
        break

vs.release()
cv2.destroyAllWindows()