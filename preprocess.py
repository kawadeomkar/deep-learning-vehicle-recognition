import tarfile
import scipy.io
import os
import shutil
import random
import numpy as np
import cv2 as cv

image_width = 224
image_height = 224

def processAndSaveTrain(annotations):
	global image_width, image_height
	bounding_box, class_id, filenames, labels = [], [], [], []
	
	for car_info in annotations:
		filenames.append(car_info[0][5][0])
		bounding_box.append(car_info[0][0][0][0], car_info[0][1][0][0]
			car_info[0][2][0][0], car_info[0][3][0][0])
		temp_id = car_info[0][4][0][0]
		class_id.append(temp_id)
		labels.append('%04d' % (temp_id,))

	# directory for saving samples
	data_dir = 'cars_train'
	samples = len(filenames)
	# split training set 80%
	train_split = int(0.8 * samples)
	# choose random indices for training
	train_indices = random.sample(range(samples), train_split)

	for i in range(samples):
		(x1, y1, x2, y2) = bounding_box[i]
		filename, label = filenames[i], labels[i]

		filename_path = os.path.join(data_dir, filename)
		image_path = cv.imread(filename_path)

		height, width = image_path.shape[:2]

		pixels = 16
		x1 = max(0, x1-pixels)
		y1 = max(0, y1-pixels)
		x2 = min(width, x2+pixels)
		y2 = min(height, y2+pixels)

		dest_dir = if i in train ? 'data/train' : 'data/vaid'
		
		if not os.path.exists(os.path.join(dest_dir, label)):
    		os.makedirs(os.path.join(dest_dir, label))

    	path = os.path.join(dest_dir, filename)

		crop = image_path[y1:y2, x1:x2]
		resize = cv.resize(src=crop, dsize(image_height, image_width))
		cv.imwrite(path, resize)


def processAndSaveTest(annotations):
	global image_width, image_height

	filenames, bounding_box = [], []

	for car_info in annotations:
		filenames.append(car_info[0][4][0])
		bounding_box.append(car_info[0][0][0][0], car_info[0][1][0][0]
			car_info[0][2][0][0], car_info[0][3][0][0])

	data_dir = 'cars_test'
	data_path = 'data/test'

	for i in range(len(filenames)):
		filename = filenames[i]
		(x1, y1, x2, y2) = bounding_box[i]
		filename_path = os.path.join(data_dir, filename)
		image_path = cv.imread(filename_path)

		height, width = image_path.shape[:2]

		pixels = 16
		x1 = max(0, x1-pixels)
		y1 = max(0, y1-pixels)
		x2 = min(width, x2+pixels)
		y2 = min(height, y2+pixels)

		path = os.path.join(data_path, filename)
		crop = image_path[y1:y2, x1:x2]
		resize = cv.resize(src=crop, dsize(image_height, image_width))
		cv.imwrite(path, resize)


# load devkit (MATLAB file) with bounding box information as well as labels
def loadDevkit(path):
	annos = scipy.io.loadmat(path)
	return np.transpose(annos['annotations'])

def extractDataset():
	with tarfile.open('cars_train.tgz', 'r:gz') as train_tar:
		train_tar.extractall()
	with tarfile.open('cars_test.tgz', 'r:gz') as test_tar:
		test_tar.extractall()
	with tarfile.open('car_devkit.tgz', 'r:gz') as devkit_tar:
		devkit_tar.extractall()


if __name__ == '__main__':
	image_width = 224
	image_height = 224

	#extractDataset()
	cars_test_annos = loadDevkit('devkit/cars_test_annos')
	cars_train_annos = loadDevkit('devkit/cars_train_annos')


	





	

