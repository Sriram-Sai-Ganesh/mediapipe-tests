# Author: Sriram Sai Ganesh
# code modified from https://github.com/google/mediapipe/blob/master/docs/solutions/holistic_model.md

import cv2
import mediapipe as mp
import numpy as np

from HelperFunctions import *


########################################
USE_WEBCAM = True 			# set to false to run model on images

# set file output location
OUTFILE_NAME = 'out/holistic_landmark_coords.txt'

# ignore landmark coords if visibility<VISIBILITY_THRESHOLD
VISIBILITY_THRESHOLD = 0.85

########################################

# define object types
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def print_landmarks(pose_landmarks, outfile = False):
	# if outfile:
	# 	outfile.write(str(pose_landmarks))
	# else:
	# 	print(pose_landmarks)
	for j in range(len(pose_landmarks)):
		print(j, '\t: ', pose_landmarks[j])


# For static images:
def holistic_on_images(IMAGE_FILES):
	BG_COLOR = (192, 192, 192) # gray
	with mp_holistic.Holistic(static_image_mode=True,	model_complexity=2, enable_segmentation=True, refine_face_landmarks=True) as holistic_model:
		for idx, file in enumerate(IMAGE_FILES):
			image = cv2.imread(file)
			image_height, image_width, _ = image.shape
			# Convert the BGR image to RGB before processing.
			pose_landmarker_result = holistic_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

			if pose_landmarker_result.pose_landmarks:
				print(
						f'Nose coordinates: ('
						f'{pose_landmarker_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
						f'{pose_landmarker_result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
				)

			annotated_image = image.copy()
			# Draw segmentation on the image.
			# To improve segmentation around boundaries, consider applying a joint
			# bilateral filter to "pose_landmarker_result.segmentation_mask" with "image".
			condition = np.stack((pose_landmarker_result.segmentation_mask,) * 3, axis=-1) > 0.1
			bg_image = np.zeros(image.shape, dtype=np.uint8)
			bg_image[:] = BG_COLOR
			annotated_image = np.where(condition, annotated_image, bg_image)
			# Draw pose, left and right hands, and face landmarks on the image.
			mp_drawing.draw_landmarks(
					annotated_image,
					pose_landmarker_result.face_landmarks,
					mp_holistic.FACEMESH_TESSELATION,
					landmark_drawing_spec=None,
					connection_drawing_spec=mp_drawing_styles
					.get_default_face_mesh_tesselation_style())
			mp_drawing.draw_landmarks(
					annotated_image,
					pose_landmarker_result.pose_landmarks,
					mp_holistic.POSE_CONNECTIONS,
					landmark_drawing_spec=mp_drawing_styles.
					get_default_pose_landmarks_style())
			cv2.imwrite('./out/annotated_' + str(idx+1) + '.png', annotated_image)
			# Plot pose world landmarks.
			my_plot_landmarks(
					pose_landmarker_result.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS, image_dimensions=(image_height, image_width))
def holistic_on_webcam(outfile=False):
	# For webcam input:
	cap = cv2.VideoCapture(0)
	with mp_holistic.Holistic(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as holistic_model:
		while cap.isOpened() and (cv2.waitKey(25) & 0xFF != ord('q')):
			success, image = cap.read()
			if not success:
				print("Ignoring empty camera frame.")
				# If loading a video, use 'break' instead of 'continue'.
				continue
			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			pose_landmarker_result = holistic_model.process(image)

			# use the generated PoseLandmarkerResult object:
			if outfile:
				outfile.write(str(pose_landmarker_result)+'\n')
				outfile.write('pose landmarks:\n')
				print_landmarks(pose_landmarker_result.pose_landmarks, outfile)
				outfile.write('face landmarks:\n')
				print_landmarks(pose_landmarker_result.face_landmarks, outfile)
				outfile.write('left_hand landmarks:\n')
				print_landmarks(pose_landmarker_result.left_hand_landmarks, outfile)
				outfile.write('right_hand landmarks:\n')
				print_landmarks(pose_landmarker_result.right_hand_landmarks, outfile)
				break

			# Draw landmark annotation on the image.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			mp_drawing.draw_landmarks(
					image,
					pose_landmarker_result.face_landmarks,
					mp_holistic.FACEMESH_CONTOURS,
					landmark_drawing_spec=None,
					connection_drawing_spec=mp_drawing_styles
					.get_default_face_mesh_contours_style())
			mp_drawing.draw_landmarks(
					image,
					pose_landmarker_result.pose_landmarks,
					mp_holistic.POSE_CONNECTIONS,
					landmark_drawing_spec=mp_drawing_styles
					.get_default_pose_landmarks_style())
			# Flip the image horizontally for a selfie-view display.
			cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
			if cv2.waitKey(5) & 0xFF == 27:
				break
	cap.release()


# print output status
outfile = open(OUTFILE_NAME, "w+")
outfile.write("Name of the file: " + outfile.name + "\n")
outfile.write("File open : " + str(not outfile.closed) + "\n")
outfile.write("Opening mode : " + str(outfile.mode) + "\n")

if USE_WEBCAM:
	holistic_on_webcam(outfile)
else:
	holistic_on_images(['./media/holistic_image.jpeg', './media/holistic_image2.jpeg', './media/face1.jpeg', './media/face2.jpeg', './media/running1.jpeg', './media/running2.jpeg'])