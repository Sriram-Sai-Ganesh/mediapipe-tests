import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
from datetime import datetime

#drawing
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


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



model_path = 'pose_landmarker_heavy.task'


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_image=0

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
	# annotated_image = draw_landmarks_on_image(mp_image, result)
	# cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
	print('pose landmarker result: {}'.format(result))

options = PoseLandmarkerOptions(
		base_options=BaseOptions(model_asset_path=model_path),
		running_mode=VisionRunningMode.LIVE_STREAM,
		result_callback=print_result)

start_time=datetime.now()

with PoseLandmarker.create_from_options(options) as landmarker:

	# create cv2 video capture object
	video_capture = cv2.VideoCapture(0)
	while video_capture.isOpened() and (cv2.waitKey(25) & 0xFF != ord('q')):
		# Capture frames.
		success, image = video_capture.read()
		# print error if no cap
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue

		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		print("running feed")

		# Get height and width of the frame.
		h, w = image.shape[:2]

		# convert cv2 image to mp image format
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
		# detect pose landmarks from the input image.
		timestamp_ms = (int)((datetime.now()-start_time).total_seconds()*1000)
		landmarker.detect_async(mp_image, timestamp_ms)

		# print resulting poseLandmarkerResult object
		# print(str(poseLandmarkerResult))

		# visualize the detection result:

		# annotated_image = draw_landmarks_on_image(image, poseLandmarkerResult)
		# cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
		
		# non-annotated image:
		cv2.imshow("Pose landmarks", image)



		# cv2.imshow("window", image)
		
		# wait 1ms between frames
		# rate is limited by webcam capture, not 1ms delay.
		cv2.waitKey(1)
