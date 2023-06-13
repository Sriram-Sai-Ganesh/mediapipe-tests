
from mediapipe.framework.formats import landmark_pb2
from typing import List, Mapping, Optional, Tuple, Union
import dataclasses
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# helper constants
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

# helper class for my_plot_landmarks override
class DrawingSpec:	
	# Color for drawing the annotation. Default to the white color.
	color: Tuple[int, int, int]
	# Thickness for drawing the annotation. Default to 2 pixels.
	thickness: int
	# Circle radius. Default to 2 pixels.
	circle_radius: int
	def __init__(self, color = WHITE_COLOR, thickness = 2, rad = 2):
		self.color=color
		self.thickness=thickness
		self.circle_radius=rad


# modifying code from https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
def my_plot_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList,
connections: Optional[List[Tuple[int, int]]] = None,
landmark_drawing_spec: DrawingSpec = DrawingSpec(
color=RED_COLOR, thickness=2),
connection_drawing_spec: DrawingSpec = DrawingSpec(
color=BLACK_COLOR, thickness=2),
elevation: int = 10,
azimuth: int = 10, image_dimensions=(3,1)):
	"""Plot the landmarks and the connections in matplotlib 3D.

	Args:
		landmark_list: A normalized landmark list proto message to be plotted.
		connections: A list of landmark index tuples that specifies how landmarks to
			be connected.
		landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
			drawing settings such as color and line thickness.
		connection_drawing_spec: A DrawingSpec object that specifies the
			connections' drawing settings such as color and line thickness.
		elevation: The elevation from which to view the plot.
		azimuth: the azimuth angle to rotate the plot.

	Raises:
		ValueError: If any connection contains an invalid landmark index.
	"""
	if not landmark_list:
		return
	plt.figure(figsize=(11, 11))
	ax = plt.axes(projection='3d')



	# scale axes to fit image aspect ratio
	from mpl_toolkits.mplot3d.axes3d import Axes3D
	from mpl_toolkits.mplot3d import proj3d
	# x_scale=0.1		# front to back
	x_scale=1		# front to back
	y_scale=1		# sideways
	z_scale=(image_dimensions[0]/image_dimensions[1])		# vertical

	scale=np.diag([x_scale, y_scale, z_scale, 1.0])
	scale=scale*(1.0/scale.max())
	scale[3,3]=1.0

	def short_proj():
		return np.dot(Axes3D.get_proj(ax), scale)

	ax.get_proj=short_proj


	ax.view_init(elev=elevation, azim=azimuth)
	
	plotted_landmarks = {}
	for idx, landmark in enumerate(landmark_list.landmark):
		if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or
				(landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
			continue
		ax.scatter3D(
				xs=[-landmark.z],
				ys=[landmark.x],
				zs=[-landmark.y],
				color=_normalize_color(landmark_drawing_spec.color[::-1]),
				linewidth=landmark_drawing_spec.thickness)
		plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
	if connections:
		num_landmarks = len(landmark_list.landmark)
		# Draws the connections if the start and end landmarks are both visible.
		for connection in connections:
			start_idx = connection[0]
			end_idx = connection[1]
			if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
				raise ValueError(f'Landmark index is out of range. Invalid connection '
				f'from landmark #{start_idx} to landmark #{end_idx}.')
			if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
				landmark_pair = [
						plotted_landmarks[start_idx], plotted_landmarks[end_idx]
				]
				ax.plot3D(xs=[landmark_pair[0][0], 
				landmark_pair[1][0]],ys=[landmark_pair[0][1], 
				landmark_pair[1][1]],zs=[landmark_pair[0][2], 
				landmark_pair[1][2]],color=_normalize_color
				(connection_drawing_spec.color[::-1]),
				linewidth=connection_drawing_spec.thickness)
	plt.show()


def _normalize_color(color):
	return tuple(v / 255. for v in color)


