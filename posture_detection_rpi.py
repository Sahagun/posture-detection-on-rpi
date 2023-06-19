import cv2
import mediapipe as mp
import pygame
import time

# Set to true if using a pi camera; false for usb camera
USING_PICAMERA = True

# Label position info
X_LABEL_START = 25
X_LABEL_END = 275
Y_LABEL = 70

# audio file 
audio_path = 'wrong.ogg'
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(audio_path)

# media pipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


if USING_PICAMERA:
	from picamera2 import Picamera2

	# Initialize the pi camera
	pi_camera = Picamera2()
	# Convert the color mode to RGB
	config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
	pi_camera.configure(config)

	# Start the pi camera and give it a second to set up
	pi_camera.start()
	time.sleep(5)
else:
	# USB Camera with openCV
	cap = cv2.VideoCapture(0)

def get_landmark_results(image):
	''' Get the landmark result from the image using mediapipe '''
	# To improve performance, optionally mark the image as not writeable to
	# pass by reference.
	image.flags.writeable = False
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = pose.process(image)

	# Draw the __pose annotation on the image.
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, results


def get_posture(image, h, w, landmarks):
	''' Get if the posture is correct or incorrect '''

	l_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
	l_hip = mp_pose.PoseLandmark.LEFT_HIP.value

	m = (calculate_slope(landmarks[l_hip].x * w, landmarks[l_hip].y * h,
						 landmarks[l_shoulder].x * w, landmarks[l_shoulder].y * h))

	posture = 'Correct'
	if -5 <= m <= 5:
		posture = 'Incorrect'

	image = write_slope((X_LABEL_START, 25), image, round(-m, 2), posture)
	return image, posture

def write_slope(org: tuple, image, slope, posture):
	''' Write on the image what the slope is '''
	# font
	font = cv2.FONT_HERSHEY_SIMPLEX
	# fontScale
	fontScale = 0.6
	# Blue color in BGR
	color = (255, 255, 255)

	# Line thickness of 2 px
	thickness = 2

	cv2.line(image, (org[0] - 2, org[1]), (X_LABEL_END, org[1]), (255, 0, 255), 50)
	cv2.line(image, (org[0] - 2, org[1]), (X_LABEL_END, org[1]), (0, 0, 0), 45)
	# Using cv2.putText() method
	image = cv2.putText(image, f'Posture Slope: {slope}', org, font,
						fontScale, color, thickness)
	# Using cv2.putText() method
	image = cv2.putText(image, f'Posture: {posture}', (org[0], org[1] + 20), font,
						fontScale, color, thickness)
	return image

def draw_image(image, landmarks, results):
	''' Draw the landmakrs on the image '''
	for idx, landmark in enumerate(landmarks):
		if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]:
			landmarks[idx].visibility = 0

	mp_drawing.draw_landmarks(image, results.pose_landmarks, 
		mp_pose.POSE_CONNECTIONS,
		mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
		mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

	return image

def resize_image(image, percent=0.5):
    h, w, _ = image.shape
    h, w = int(h * percent), int(w * percent)
    image = cv2.resize(image, (w, h))
    return image, h, w

def calculate_slope(x1, y1, x2, y2):
    # print(x1, y1, x2, y2)
    m = (y2 - y1) / (x2 - x1)
    return m

def process_image(image):
	''' process the image '''
	h, w, _ = image.shape
	image, results = get_landmark_results(image)
	
	
	landmarks = results.pose_landmarks.landmark
	
	image, posture = get_posture(image, h, w, landmarks)

	if posture == 'Incorrect':
		print('playing incorrect sound...')
		pygame.mixer.music.play()

		while pygame.mixer.music.get_busy():
			continue
		
	image = draw_image(image, landmarks, results)	

	return image



while True:
	# Capture frame-by-frame	
	if USING_PICAMERA:
		image = pi_camera.capture_array()
	else:
		_, image = cap.read()
		
	# Resize the image to process faster
	image, h, w = resize_image(image, percent=0.5)


	try:
		image = process_image(image)
	except Exception as e:
		print(e)
									
	cv2.imshow('pose', image)
	
	# Press Q on keyboard to  exit
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()


