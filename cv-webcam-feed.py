# Author: Sriram Sai Ganesh

# Guides consulted: 
# https://learnopencv.com

import cv2

# for webcam input, replace file name with 0.
file_name = 0

# create cv2 video capture object
video_capture = cv2.VideoCapture(file_name)

# get vid capture metadata
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)

# create VideoWriter object to write to .mp4 file
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Video writer
# video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

# continue while cap is open and 'Q' is not pressed
while video_capture.isOpened() and (cv2.waitKey(25) & 0xFF != ord('q')):
	# Capture frames.
	success, image = video_capture.read()
	# print error if no cap
	if not success:
		print("ERROR: Null.Frames")
		break

	# Get fps.
	fps = video_capture.get(cv2.CAP_PROP_FPS)

	print(str(width)+" x "+str(height)+"  -  "+str(fps)+" fps")

	# Get height and width of the frame.
	h, w = image.shape[:2]

	# detect pose landmarks from the input image.
	# detection_result = detector.detect(image)

	cv2.imshow("window", image)
	
	# wait 1ms between frames
	# rate is limited by webcam capture, not 1ms delay.
	cv2.waitKey(1)

	'''
	Meeting notes: 
	ischemic stroke and heamorrage
	focus is ischemic
	nih stroke scale, 32 points analyzed on 15 different things
	levels of stroke	
	'''