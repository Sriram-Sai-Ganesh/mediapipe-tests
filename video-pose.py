# Author Sriram Sai Ganesh
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
from datetime import datetime

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from HelperFunctions import *

# debug: detect location of prints by appending traceback to any stdouts
import sys
import traceback

class TracePrints(object):
	def __init__(self):    
		self.stdout = sys.stdout
	def write(self, s):
		self.stdout.write("Writing %r\n" % s)
		traceback.print_stack(file=self.stdout)
# sys.stdout = TracePrints()

# model returns these points:
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

pose_landmarker_dict = {'NOSE' : 0, 'LEFT_EYE_INNER' : 1, 'LEFT_EYE' : 2, 'LEFT_EYE_OUTER' : 3, 'RIGHT_EYE_INNER' : 4, 'RIGHT_EYE' : 5, 'RIGHT_EYE_OUTER' : 6, 'LEFT_EAR' : 7, 'RIGHT_EAR' : 8, 'LEFT_MOUTH' : 9, 'RIGHT_MOUTH' : 10, 'LEFT_SHOULDER' : 11, 'RIGHT_SHOULDER' : 12, 'LEFT_ELBOW' : 13, 'RIGHT_ELBOW' : 14, 'LEFT_WRIST' : 15, 'RIGHT_WRIST' : 16, 'LEFT_PINKY' : 17, 'RIGHT_PINKY' : 18, 'LEFT_INDEX' : 19, 'RIGHT_INDEX' : 20, 'LEFT_THUMB' : 21, 'RIGHT_THUMB' : 22, 'LEFT_HIP' : 23, 'RIGHT_HIP' : 24, 'LEFT_KNEE' : 25, 'RIGHT_KNEE' : 26, 'LEFT_ANKLE' : 27, 'RIGHT_ANKLE' : 28, 'LEFT_HEEL' : 29, 'RIGHT_HEEL' : 30, 'LEFT_FOOT_INDEX' : 31, 'RIGHT_FOOT_INDEX' : 32
}

# objects
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# function to draw landmarks on video feed per-frame
# receives point locations from PoseLandmarker model
# draws points using the mediapipe solutions' drawingutils
def draw_landmarks_on_image(rgb_image, detection_result):
	pose_landmarks_list = detection_result.pose_landmarks
	annotated_image = np.copy(rgb_image)

	# Loop through the detected poses to visualize.
	for idx in range(len(pose_landmarks_list)):
		pose_landmarks = pose_landmarks_list[idx]

		# Draw the pose landmarks.
		pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		pose_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
		])
		solutions.drawing_utils.draw_landmarks(
			annotated_image,
			pose_landmarks_proto,
			solutions.pose.POSE_CONNECTIONS,
			solutions.drawing_styles.get_default_pose_landmarks_style())
	return annotated_image

# function to print PoseLandmarkerResult object
def print_pose_landmarker_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
	print('pose landmarker result: {}'.format(result))

def plot_pose_landmarker_variables(result: mp.tasks.vision.PoseLandmarkerResult):
	print(get_pose_landmarker_coordinates(result, "LEFT_ANKLE"))

def get_pose_landmarker_coordinates(result, *args):
	for s in args:
		print(get_coords_for_landmark(result, s))

def get_coords_for_landmark(result, location):
	if location not in pose_landmarker_dict:
		return (-1, -1, -1)
	return (result.pose_landmarks[0][0].x, result.pose_landmarks[0][0].y, result.pose_landmarks[0][0].z)

	print(str(result.pose_landmarks[0][0].x) + " " + str(result.pose_landmarks[0][0].y))

# calculate angle between horizontal and individual's spine.
def calculate_spine_angle(result: mp.tasks.vision.PoseLandmarkerResult):
	# ASSUMPTION -- floor is completely level with camera
	# TODO: detect floor plane, and account for that angle when calculating spine angle.
	
	
	# trig goes here
	print("<calc spine angle>")

# set model path
model_path = 'models/pose_landmarker_heavy.task'
# i/o streams
video_capture = cv2.VideoCapture('./media/video.mp4')
outfile_name = "out/landmark_coords.txt"


# name is filename for mediapipe model
model_name=model_path.split("/")[-1].split(".")[0]
print("USING MODEL: "+model_name)

# Create a pose landmarker instance with the video running mode:
options = PoseLandmarkerOptions(
		base_options=BaseOptions(model_asset_path=model_path),
		running_mode=VisionRunningMode.VIDEO)

start_time=datetime.now()

outfile = open(outfile_name, "w+")
outfile.write("Name of the file: " + outfile.name)
outfile.write("Closed or not : " + str(outfile.closed))
outfile.write("Opening mode : " + str(outfile.mode))

# with landmarker:
landmarker_model = PoseLandmarker.create_from_options(options)

# run until video cap is closed, or 'q' is pressed.
while video_capture.isOpened() and (cv2.waitKey(25) & 0xFF != ord('q')):
	# Capture frames.
	success, image = video_capture.read()
	# print error if no cap
	if not success:
		print("ERROR: Null.Frames")
		break

	# Get height and width of the frame.
	# h, w = image.shape[:2]

	# convert cv2 image to mp image format
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
	# detect pose landmarks from the input image.
	timestamp_ms = (int)((datetime.now()-start_time).total_seconds()*1000)
	pose_landmarker_result = landmarker_model.detect_for_video(mp_image, timestamp_ms)

	# use the generated PoseLandmarkerResult object:
	# print(str(poseLandmarkerResult))
	# print(calculate_spine_angle(pose_landmarker_result))

	# visualize the detection result:
	plot_pose_landmarker_variables(pose_landmarker_result)
	annotated_image = draw_landmarks_on_image(image, pose_landmarker_result)
	cv2.imshow(model_name,cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

	# wait 1ms between frames (not a bottleneck)
	cv2.waitKey(1)




































