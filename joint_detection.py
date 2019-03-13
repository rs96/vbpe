# full imports
import numpy as np #version 1.15.4 doesn't have the weird error
import cv2
# partial imports
from random import shuffle, random
from operator import itemgetter
from time import clock
from math import sqrt
# neural network imports
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU
from keras.utils import to_categorical
from scipy.io import loadmat


# this is a list of window sizes related to each joint
window_sizes = [14,16,18]#[15,12,12,12,12,12,12,12,12,12,12,12,12,12]
# threshold value for probabilty of joint being detected
probability_threshold = 0.75

# this is where i build my models
def build_model_right_wrist():
	model = Sequential()
	model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(42,42,3)))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(64, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(12, init='uniform', activation='relu'))
	model.add(Dense(4, init='uniform', activation='relu'))
	model.add(Dense(1, activation='sigmoid')) # sigmoid?
	return model

def build_model_head_top():
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(42,42,3)))
	model.add(Dropout(rate=0.5))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(rate=0.5))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(16, activation='linear', kernel_initializer='uniform'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(8, activation='linear', kernel_initializer='uniform'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(4, activation='linear', kernel_initializer='uniform'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(1, activation='sigmoid'))
	return model

def train_and_evaluate_model(model, x_train, x_test, y_train, y_test):
	# definition of the learning process, how the the values be changed etc.
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	# Convert labels to categorical one-hot encoding
	#y_train = to_categorical(y_train, num_classes=1)
	#y_test = to_categorical(y_test, num_classes=1)

	# Train the model, iterating on the data in batches of 32 samples
	model.fit(x_train, y_train, batch_size=64, epochs=64, verbose=2)

	# Evaluate the model
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss: ',score[0])
	print('Test accuracy: ',score[1])
    
def prepare_image_train(image):
	image = cv2.resize(image,(42,42))
	return image
	
def prepare_image_test(image):
    
    #th = 200
    #h,w,_ = image.shape
    #tw = int(round(w*(th/h)))
    #image = cv2.resize(image,(tw,th), interpolation=cv2.INTER_AREA)
    
    return image

def prepare_image_tile(image):
	image = cv2.resize(image,(42,42))
	image = np.expand_dims(image,axis=0)
	return image
	
def set_window_sizes(sizes):
    global window_sizes
    window_sizes = []
    
    for ws in sizes:
        window_sizes.append(ws)
    
    return True

def get_data(chosen_joint):
	# gather the data and load it in
	# put into training and test data
	# image data
	images = images_to_numpy(1,2000)
	# joint information
	# 3x14x2000, i will get 2000x3 as that's better info and not stupid
	joints = []
	joint_data = loadmat('joints.mat')['joints']
	for i in range(2000):
		joints.append([joint_data[0][chosen_joint][i],joint_data[1][chosen_joint][i],joint_data[2][chosen_joint][i]])

	# [x,y,visibility - 1=obscured, 0=visible (how silly)]
	joints = np.array(joints)

	return images, joints


# function for getting images as numpy arrays
def images_to_numpy(a,b):
	image_array = []

	for i in range(a,b+1):
		if(i<10):
			i_str = '000'+str(i)
		elif(i<100):
			i_str = '00'+str(i)
		elif(i<1000):
			i_str = '0'+str(i)
		else:
			i_str = str(i)
		image=cv2.imread('images/im'+i_str+'.jpg') #file_path+
		#prepare image for CNN
		#image=prepare_image(image)
		image_array.append(image)
		#if(i%50==0):
		#	print(i)

	image_array = np.array(image_array)

	return image_array

def increase_dataset(training_data):

	for i in range(len(training_data)):
		image = training_data[i][0]

		reversed_image = cv2.flip(image,1)
		upside_down = cv2.flip(image,0)
		upside_down_reversed = cv2.flip(image,-1)

		training_data.append([reversed_image,training_data[i][1]])
		training_data.append([upside_down,training_data[i][1]])
		training_data.append([upside_down_reversed,training_data[i][1]])

		# required for rotation
		rows,cols,_ = image.shape

		for angle in [10,-10,20,-20,30,-30]:
			rotation_matrix = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)),angle,1.2)
			
			rotated = cv2.warpAffine(image,rotation_matrix,(cols,rows))
			reversed_image_rotated = cv2.warpAffine(reversed_image,rotation_matrix,(cols,rows))
			upside_down_rotated = cv2.warpAffine(upside_down,rotation_matrix,(cols,rows))
			upside_down_reversed_rotated = cv2.warpAffine(upside_down_reversed,rotation_matrix,(cols,rows))

			training_data.append([rotated,training_data[i][1]])
			training_data.append([reversed_image_rotated,training_data[i][1]])
			training_data.append([upside_down_rotated,training_data[i][1]])
			training_data.append([upside_down_reversed_rotated,training_data[i][1]])

	return training_data


# splits and sorts training/test data and give it back as model wants it
def split_data(data):
    
	# randomise the data ordering to stop any trends
	shuffle(data)

	split=0.95  # what amount of data to be training data
	training_size=int(len(data)*split)
	testing_size=len(data)-training_size

	y_train = []
	x_train = []
	for i in range(training_size):
		y_train.append(data[i][1])
		x_train.append(data[i][0])
	y_train = np.array(y_train)
	x_train = np.array(x_train)

	y_test = []
	x_test = []
	for i in range(len(data)-1,training_size+1,-1):
		y_test.append(data[i][1])
		x_test.append(data[i][0])
	y_test = np.array(y_test)
	x_test = np.array(x_test)

	return x_train, x_test, y_train, y_test

#tile_h,tile_w are height and width output tiles
def tile_image(image,tile_h,tile_w):
	x_step = 1 #int(tile_w/5)
	y_step = 1 #int(tile_h/5)
	tiles = []
	coords = []
	for i in range(int(tile_h/2),image.shape[0]-int(tile_h/2),y_step):	# each row
		for j in range(int(tile_w/2),image.shape[1]-int(tile_w/2),x_step): # each column
			if(image[i:i+tile_h,j:j+tile_w].shape!=(tile_h,tile_w,3)):
				continue
			#tiles=np.append(tiles,image[i:i+tile_h,j:j+tile_w],axis=1)
			tiles.append(image[i:i+tile_h,j:j+tile_w,:3])
			coords.append([i+int(tile_h/2),j+int(tile_w/2)])
	tiles=np.array(tiles)
	return tiles,coords

def get_joint_hits(images,joints,chosen_joint):

	#window_size = window_sizes[chosen_joint]

	joint_images = []
	
	for ws in range(len(window_sizes)):
		
		window_size = window_sizes[ws]

		if(len(images)!=len(joints)):
			print('mismatch of joints and images!')
			print('images:',len(images))
			print('joints:',len(joints))
	
		for i in range(len(images)):
			if(joints[i][2]==0.0):
				joint_x = int(round(joints[i][0]))
				joint_y = int(round(joints[i][1]))
				xd = 0
				yd = 0
				if(joint_x-window_size<0):
					x1 = 0
					xd = abs(joint_x-window_size)
				else:
					x1 = joint_x-window_size
				x2 = min(joint_x+window_size,images[i].shape[1])+xd
				if(joint_y-window_size<0):
					y1 = 0
					yd = abs(joint_y-window_size)
				else:
					y1 = joint_y-window_size
				y2 = min(joint_y+window_size,images[i].shape[0])+yd
				joint_images.append(images[i][y1:y2,x1:x2])
				#print(int(joints[i][0]))

	return joint_images

def get_joint_misses(images,joints,chosen_joint):

	not_joint_images = []
	#window_size = window_sizes[chosen_joint]
	
	for ws in range(len(window_sizes)):
		window_size = window_sizes[ws]

		if(len(images)!=len(joints)):
			print('mismatch of joints and images!')
			print('images:',len(images))
			print('joints:',len(joints))

		for i in range(len(images)):
			bad_choice = True
			joint_x = int(round(joints[i][0]))
			joint_y = int(round(joints[i][1]))
			while(bad_choice):
				random_y = int(images[i].shape[0]*random())
				random_x = int(images[i].shape[1]*random())
				# for checking the other image doesn't accidentally include a wrist in it
				# this may need to be expanded to include left wrists, but shouldn't make a big difference to start with
				if((abs(random_x-joint_x)<window_size) and (abs(random_y-joint_y)<window_size)):
					bad_choice = True
					#print('test 1')
				#elif(abs(random_y-joint_y)<window_size):
				#	bad_choice = True
				#	#print('test 2')
				elif((random_y-window_size)<0 or (random_y+window_size)>images[i].shape[0]):
					bad_choice = True
				#print('test 3')
				elif((random_x-window_size)<0 or (random_x+window_size)>images[i].shape[1]):
					bad_choice = True
					#print('test 4')
				else:
					not_joint_images.append(images[i][random_y-window_size:random_y+window_size,random_x-window_size:random_x+window_size])
					bad_choice = False

	return not_joint_images

def get_images_with_visible_joint(images,joints,chosen_joint): #chosen_joint is the joint number in question

	if(len(images)!=len(joints)):
		print('images and joints data size mismatch')

	for i in range(len(images)):
		if(joints[2]==0.0):
			images_visible.append(images[i])
			joints_visible.append(joints[i])
			

	return images_visible,joints_visible

def confidence_map(model,image):

	confidence_map = [] # [x,_pos,y_pos,certainty]

	image_h = image.shape[0]
	image_w = image.shape[1]
	#window_size = window_sizes[chosen_joint]
	joints_list = []
	
	for ws in range(len(window_sizes)):
		window_size = window_sizes[ws]

		# iterate through all possible sliding window sizes and locations
		for y in range(window_size,image_h-window_size):
			for x in range(window_size,image_w-window_size):
				window = image[y-window_size:y+window_size,x-window_size:x+window_size]
				window_for_prediction = prepare_image_tile(window)
				prediction = model.predict(window_for_prediction)
				#print(prediction)
				#if(prediction[0][0]>0.9):
					#print(prediction[0][0])
				confidence_map.append([x,y,prediction[0][0]])

	confidence_map = sorted(confidence_map, key=itemgetter(2))
	confidence_map = confidence_map[-25:]

	return confidence_map

def get_heatmap(model,images,im_displacements):

	heat_map = np.zeros((1080,1920),dtype=np.uint8) # this is where i will colour my confidences
	# size may need to be set in a set_frame_size() method
	hottest_value = 0 # stores the most likely value there is
	
	# TO DO: IF IMAGES ARE WHOLE FRAMES ONLY NEED TO DO IT ONCE
	if(len(images)==2):
		frame_w = 1920
		frame_h = 1080
		if((images[0].shape[1]==frame_w) and (images[1].shape[1]==frame_w) and (images[0].shape[0]==frame_h) and (images[1].shape[0]==frame_h)):
			images = [images[0]] # don't need second image as they are the same thing, i.e the original frame, halves first processing time
	
	for im_index in range(len(images)):
		joints_list = [] # structure of potential joint [x,y,probability]
		image_h = images[im_index].shape[0]
		image_w = images[im_index].shape[1]
		coords_to_check = [] # each coord = [x,y]
		
		#print('displacement',im_index,':',im_displacements[im_index])

		#im_index = images.index(image.all())

		for ws in range(len(window_sizes)):
			window_size = window_sizes[ws]
			coords_to_check.append([])
			if(window_size*2>min(image_h,image_w)): # this MUST be the same as [gotoblock further down]
				continue

			# quick scan through image
			for y in range(window_size,image_h-window_size,window_size):
				for x in range(window_size,image_w-window_size,window_size):
					window = images[im_index][y-window_size:y+window_size,x-window_size:x+window_size]
					window_for_prediction = prepare_image_tile(window)
					if(model.predict(window_for_prediction)>probability_threshold):
						for yd in range(y-window_size,y+window_size+1,3): # 9x faster with step 3
							for xd in range(x-window_size,x+window_size+1,3):
								if(0<yd-window_size and yd+window_size<image_h and 0<xd-window_size and xd+window_size<image_w):
									coords_to_check[ws].append([xd,yd])

		# iterate through all sliding window potential areas
		for ws in range(len(window_sizes)):
			window_size = window_sizes[ws]
			if(window_size>min(image_h,image_w)): # this bit otherwise might have error
				continue
			#print(len(coords_to_check[ws]))
			for c in coords_to_check[ws]:
				window = images[im_index][c[1]-window_size:c[1]+window_size,c[0]-window_size:c[0]+window_size]
				window_for_prediction = prepare_image_tile(window)
				prediction = model.predict(window_for_prediction)
				if(prediction[0][0]>probability_threshold):
					colour = int(255*((prediction[0][0]-probability_threshold)*(1/(1-probability_threshold)))**1.5)
					if(colour>hottest_value):
						hottest_value = colour
					#heat_map[c[1],c[0]] = colour
					for i in range(c[1]-1,c[1]+2):# 3x3 square coloured
						for j in range(c[0]-1,c[0]+2): # not inclusive lists!
							if(colour>heat_map[i+im_displacements[im_index][1],j+im_displacements[im_index][0]]):
								heat_map[i+im_displacements[im_index][1],j+im_displacements[im_index][0]] = colour

	#cv2.imshow('display',np.hstack((image,heat_map)))
	#for im in range(len(images)):
	#	cv2.imshow('frame',images[im])
	#cv2.imshow('heat_map',heat_map)
	#cv2.waitKey()
	cv2.imwrite('heat_map.png',heat_map)
	#cv2.destroyAllWindows()
	
	return heat_map, hottest_value

def get_k_centroids(heat_map,hottest_value,k):

	centroids = []
	largest_contours = []
	thresh_value = 75

	#if(hottest_value<thresh_value):
	#	thresh_value = (hottest_value*0.85) # minimises times when no joint is found

	_,hm_thresh = cv2.threshold(heat_map,thresh_value,255,cv2.THRESH_BINARY)

	#cv2.imshow('thresh',hm_thresh)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	contours,_ = cv2.findContours(hm_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

	contours = sorted(contours, key=lambda x: cv2.contourArea(x))
	
	# NEED TO CHECK SURFACE AREA/PERIMETER RATIO - STOP LONG LINES BEING PICKED
	
	# ROT RECT TO SEE WHAT IS LONG AND THIN, SQUARER OBJECTS ARE MORE REALISTIC
	
	
	'''
	for c in contours:
		a = cv2.contourArea(c)
		p = cv2.arcLength(c,True)
		print('perimeter:',p)
		print('area:',a)
		print('ratio:',a/p)
	'''
	largest_contours = contours[-k:]
	
	cv2.drawContours(hm_thresh,largest_contours,-1,200,1)
	
	#cv2.imwrite('latesthm.jpg',hm_thresh)
	
	#try:
	for i in range(k):
		M = cv2.moments(largest_contours[i])
		centroids.append([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
	#except:
	#	for i in range(k):
	#		centroids.append([None,None])

	#cv2.imshow('thresh',hm_thresh)
	cv2.imwrite('thresh.png',hm_thresh)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return centroids

def get_neck_centroid(heat_map,hottest_value,head_top_position):

	centroids = []
	largest_contours = []
	thresh_value = 75

	print('head top position:',head_top_position)

	#if(hottest_value<thresh_value):
	#	thresh_value = (hottest_value*0.85) # minimises times when no joint is found

	_,hm_thresh = cv2.threshold(heat_map,thresh_value,255,cv2.THRESH_BINARY)

	contours,_ = cv2.findContours(hm_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

	contours = sorted(contours, key=lambda x: cv2.contourArea(x))

	k = min(len(contours),5)

	largest_contours = contours[-k:]
	for i in range(k):
		M = cv2.moments(largest_contours[i])
		centroids.append([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])

	# get distance to he_top and order them from closest to furthest
	centroids = sorted(centroids, key=lambda x: distance_between_points(head_top_position,x))
	centroids = [centroids[0]]
	
	cv2.drawContours(hm_thresh,largest_contours,-1,200,1)

	#cv2.imshow('thresh',hm_thresh)
	cv2.imwrite('thresh.png',hm_thresh)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return centroids

def distance_between_points(a,b):
	xd = a[0]-b[0]
	yd = a[1]-b[1]
	d = sqrt(xd**2 + yd**2)
	#d = np.linalg.norm(a-b)
	return d
	
def get_potential_joints(model,image):

	joints_list = [] # structure of potential joint [x,y,probability]
	image_h = image.shape[0]
	image_w = image.shape[1]

	for ws in range(len(window_sizes)):
		window_size = window_sizes[ws]
		
		# iterate through all possible sliding window sizes and locations
		for y in range(window_size,image_h-window_size):
			for x in range(window_size,image_w-window_size):
				window = image[y-window_size:y+window_size,x-window_size:x+window_size]
				window_for_prediction = prepare_image_tile(window)
				prediction = model.predict(window_for_prediction)
				joints_list.append([x,y,prediction[0][0]])

	joints_list = sorted(joints_list, key=itemgetter(2))
	joints_list = joints_list[-25:]

	return joints_list

# is about 10x quicker
def get_potential_joints_fast(model,image):

	joints_list = [] # structure of potential joint [x,y,probability]
	image_h = image.shape[0]
	image_w = image.shape[1]
	coords_to_check = [] # each coord = [x,y]
	for ws in range(len(window_sizes)):
		window_size = window_sizes[ws]

		coords_to_check.append([])

		# quick scan through image
		for y in range(window_size,image_h-window_size,window_size):
			for x in range(window_size,image_w-window_size,window_size):
				window = image[y-window_size:y+window_size,x-window_size:x+window_size]
				window_for_prediction = prepare_image_tile(window)
				if(model.predict(window_for_prediction)>probability_threshold):
					for yd in range(y-window_size,y+window_size):
						for xd in range(x-window_size,x+window_size):
							if(0<yd-window_size and yd+window_size<image_h and 0<xd-window_size and xd+window_size<image_w):
								coords_to_check[ws].append([xd,yd])

	# iterate through all possible sliding window sizes and locations
	for ws in range(len(window_sizes)):
		window_size = window_sizes[ws]
		for c in coords_to_check[ws]:
			window = image[c[1]-window_size:c[1]+window_size,c[0]-window_size:c[0]+window_size]
			window_for_prediction = prepare_image_tile(window)
			prediction = model.predict(window_for_prediction)
			joints_list.append([c[0],c[1],prediction[0][0]])

	joints_list = sorted(joints_list, key=itemgetter(2))
	joints_list = joints_list[-25:] #int(len(joints_list)/100)

	return joints_list
    
def get_average_positions(joints_list):

	average_positions = [] #each position as an item: [x,y]

	for i in range(len(joints_list)):

		joints_list[i] = sorted(joints_list[i],key=itemgetter(0,1))

		if(i<2): #i<2

			# find the weighted mean (weight is probability) of the most probable points
			xa = 0
			ya = 0

			for j in range(len(joints_list[i])):
				xa += joints_list[i][j][0]#*joints_list[i][j][2]
				ya += joints_list[i][j][1]#*joints_list[i][j][2]
			    
			xa = int(round(xa/len(joints_list[i])))
			ya = int(round(ya/len(joints_list[i])))
			
			average_positions.append([xa,ya])

		else:

			grouping_index = 0
			max_dist = 0

			for j in range(len(joints_list[i])-1):
				if(max_dist<(((joints_list[i][j][0]-joints_list[i][j+1][0]))**2+((joints_list[i][j][1]-joints_list[i][j+1][1])**2))):
					max_dist = (((joints_list[i][j][0]-joints_list[i][j+1][0]))**2+((joints_list[i][j][1]-joints_list[i][j+1][1])**2))
					grouping_index = j + 1

			xs = []
			ys = []

			for j in range(grouping_index):
				for k in range((int(100*((joints_list[i][j][2]-probability_threshold)*2)**3))):
					xs.append(joints_list[i][j][0])#*joints_list[i][j][2]
					ys.append(joints_list[i][j][1])#*joints_list[i][j][2]

			try:
				xs = int(round(np.mean(xs)))
				ys = int(round(np.mean(ys)))
			except:
				print(xs)
				print(ys)

			average_positions.append([xs,ys])

			xs = []
			ys = []

			for j in range(grouping_index,len(joints_list[i])):
				for k in range(int(100*((joints_list[i][j][2]-probability_threshold)*2)**3)): #(100*(joints_list[i][j][2])**4)
					xs.append(joints_list[i][j][0])#*joints_list[i][j][2]
					ys.append(joints_list[i][j][1])#*joints_list[i][j][2]

			try:
				xs = int(round(np.mean(xs)))
				ys = int(round(np.mean(ys)))
			except:
				print(xs)
				print(ys)

			average_positions.append([xs,ys])

	return average_positions

def load_models():

	models = []

	print('Loading Head Top Model')
	models.append(load_model('models/head_top_detector_v1.h5')) # seems to work well
	print('Loading Neck Model')
	models.append(load_model('models/neck_detector_v1.h5')) #average to poor
	'''
	print('Loading Shoulder Model')
	models.append(load_model('models/shoulder_detector_v1.h5')) #good for clear calls
	print('Loading Elbow Model')
	models.append(load_model('models/elbow_detector_v1.h5'))
	print('Loading Wrist Model')
	models.append(load_model('models/wrist_detector_v1.h5'))	#rubbish!
	print('Loading Hip Model')
	models.append(load_model('models/hip_detector_v1.h5'))
	print('Loading Knee Model')
	models.append(load_model('models/knee_detector_v1.h5'))
	print('Loading Ankle Model')
	models.append(load_model('models/ankle_detector_v1.h5'))
	print('Models Loaded!')
	'''

	return models

def train():

	chosen_joints = [7,10] # left and right wrists
	
	training_data = [] # [image,label]
	
	for chosen_joint in chosen_joints:
		images, joints = get_data(chosen_joint)
		joint_images = get_joint_hits(images,joints,chosen_joint)
		not_joint_images = get_joint_misses(images,joints,chosen_joint)

		for i in range(len(joint_images)):
			training_data.append([joint_images[i],1])
			training_data.append([not_joint_images[i],0])

	# artificially increase the training data size
	training_data = increase_dataset(training_data)
	#training_data = training_data[:10000] # dataset size restriction used for quick testing

	for i in range(len(training_data)):
		training_data[i][0] = prepare_image_train(training_data[i][0])
		
	print('Size of Training Data: ',len(training_data))
	
	x_train,x_test,y_train,y_test = split_data(training_data)
	
	model = build_model_head_top()
	
	train_and_evaluate_model(model,x_train,x_test,y_train,y_test)
	
	model.save('elbow_detector.h5')
	
	return True

def run_model():

	chosen_joint = 0

	print('Detecting Ankles')
	model = load_model('models/head_top_detector_v1.h5')

	example = cv2.imread('charlotte10.png',1)
	
	example = prepare_image_test(example)
	
	s = clock()
	joints_list = get_potential_joints_fast(model,example)
	# time to beat: 18.1 - 18.25
	print(clock()-s)

	#cm = confidence_map(model,example)

	#coloured_image = colour_image_confidence(example,cm)

	neck = get_average_positions([joints_list])[0]

	print(neck)

	cv2.circle(example,(neck[0],neck[1]),3,[0,0,255],-1)

	cv2.imwrite('haha.jpg',example)

	return True


def run_video():

	c = 0

	models = load_models()

	cap = cv2.VideoCapture('walking_short.mp4')


	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #*'DIVX'
	out = cv2.VideoWriter('annotated_walking.avi',fourcc, 10, (1080,1920))


	while(c<100):

		c += 1

		if(c%1!=0):
			continue

		_,frame = cap.read()

		print('current frame:',c)


		# preprocess frame
		#frame = prepare_image_test(frame)

		#joints_list = []

		#for model in models:
		#	joints_list.append(get_potential_joints(model,frame))
			
		#joints_list = get_average_positions(joints_list)

		#j_numbering = [0,1,2,2,3,3,4,4,5,5,6,6,7,7]
		#for i in range(len(joints_list)):
		#    cv2.putText(frame,str(j_numbering[i]),(joints_list[i][0],joints_list[i][1]),cv2.FONT_HERSHEY_PLAIN,0.5,[0,0,255],1)

		out.write(frame)


	cap.release()
	out.release()

	return True

def get_joints(models,frame):

	#[Right ankle, Right knee, Right hip, Left hip, Left knee, Left ankle,
	#Right wrist, Right elbow, Right shoulder, Left shoulder, Left elbow, Left wrist, Neck, Head top]
	joints_list = []

	#work down the body as that seems sensible and easy to think about (numbers are indexes in the joints_list)
	#
	#		0
	#     |   |
	#		1
	#	2-------2
	#  /	|	 \
	# 3	   5 5	  3
	# |	  /   \   |
	# 4   6   6   4
	#     |   |
	#     7   7

	for model in models:
		joints_list.append(get_potential_joints(model,frame))

	joints_list = get_average_positions(joints_list)

	return joints_list

#train()
#run_model()
#run_video()

#model = load_model('models/head_top_detector_v1.h5')

#image = cv2.imread('charlotte10.png',1)
#print(get_k_centroids(get_heatmap(model,image),1))


'''
images, joints = get_data(13)
for i in range(len(joints)):
    print(get_k_centroids(get_heatmap(model,prepare_image_test(images[i])),1))
    print('actual:',joints[i])
'''

'''
print('start')
frame = cv2.imread('charlotte10.png',1)
frame = prepare_image_test(frame)
print('image loaded')
models = load_models()
joints_list = get_joints(models,frame)
j_numbering = [0,1,2,2,3,3,4,4,5,5,6,6,7,7]
for i in range(len(joints_list)):
	cv2.putText(frame,str(j_numbering[i]),(joints_list[i][0],joints_list[i][1]),cv2.FONT_HERSHEY_PLAIN,0.5,[0,0,255],1)
cv2.imwrite('full body.png',frame)
'''



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################