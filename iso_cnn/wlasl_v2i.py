# Importing all necessary libraries
import cv2
import os
from tqdm import tqdm


root_path = '../../WLASL2000'
save_root = '../../WLASL2000_imgs'
for v_f in tqdm(os.listdir(root_path)):
	key = v_f.replace('mp4','')
	# Read the video from specified path
	cam = cv2.VideoCapture(os.path.join(root_path, v_f))

	save_path = os.path.join(save_root, key)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# frame
	currentframe = 0

	while(True):

		# reading from frame
		ret,frame = cam.read()

		if ret:
			# if video is still left continue creating images
			file_name = str(currentframe) + '.jpg'
			name = os.path.join(save_path, file_name)

			# writing the extracted images
			cv2.imwrite(name, frame)

			# increasing counter so that it will
			# show how many frames are created
			currentframe += 1
		else:
			break

	# Release all space and windows once done
	cam.release()
	cv2.destroyAllWindows()