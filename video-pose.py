import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
from datetime import datetime

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


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
def print_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
	print('pose landmarker result: {}'.format(result))

# calculate angle between horizontal and individual's spine.
def calculate_spine_angle(result: mp.tasks.vision.PoseLandmarkerResult):
	# ASSUMPTION -- floor is completely level with camera
	# TODO: detect floor plane, and account for that angle when calculating spine angle.
	
	
	# trig goes here
	print()


# set model path
model_path = 'models/pose_landmarker_heavy.task'
# name is filename for mediapipe model
model_name=model_path.split("/")[-1].split(".")[0]
print("MODEL NAME IS "+model_name)

# Create a pose landmarker instance with the video running mode:
options = PoseLandmarkerOptions(
		base_options=BaseOptions(model_asset_path=model_path),
		running_mode=VisionRunningMode.VIDEO)

start_time=datetime.now()

# with landmarker:
landmarker_model = PoseLandmarker.create_from_options(options)

# create cv2 video capture object
video_capture = cv2.VideoCapture('video.mp4')
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
	print(calculate_spine_angle(pose_landmarker_result))

	# visualize the detection result:

	annotated_image = draw_landmarks_on_image(image, pose_landmarker_result)
	cv2.imshow(model_name,cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

	# wait 1ms between frames (not a bottleneck)
	cv2.waitKey(1)
