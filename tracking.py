# full imports
import numpy as np #version 1.15.4 doesn't have the weird error, use 'pip install --upgrade numpy'
import cv2
import joint_detection as jd
# partial imports
from random import shuffle, random
from operator import itemgetter
from time import clock
from math import sqrt
# neural network imports
from keras.models import load_model


#image = cv2.imread('charlotte10.png',1)
#print(jd.get_k_centroids(jd.get_heatmap(model,image),1))

# variables
current_joint_contours = []

def play_video_with_locations(l):

    c = 0
    cap = cv2.VideoCapture('walking2.mp4')

    while(True):
        input('press enter to play video')
        while(c<173):

            try:
                _,frame = cap.read()
            except:
                break

            cv2.circle(frame,(l[c][0],l[c][1]),3,[0,0,255],-1)
            cv2.imshow('frame',frame)
            cv2.waitKey(24)

            c += 1

    cv2.destroyAllWindows()

    return True

def mask_frame(frame,joint):

    frame_masked = frame.copy()

    for i in range(14):
        if(i!=joint):
            #contours_to_draw = current_joint_contours[:]
            cv2.drawContours(frame_masked,[current_joint_contours[i]],-1,255,4)

    return frame_masked

def video_track():
    
	# first two joints (head top and neck) used to draw box for joint finding

	# model to be used
	numofjoints = [[0],[0]] #[0,1] # [will be a full list soon]
	models = []
	models.append(load_model('models/head_top_detector_v1.h5'))
	models.append(load_model('models/neck_detector_v1.h5'))
	#models.append(models[0]) # only temporary for two joint testing
	locations = [None,None] # items in the form [x,y]
	speeds = [] # items in the form of s
	velocities = [] # items in the form of [vx,vy]
    
	cap = cv2.VideoCapture('walking.mp4')
    
	base_size = 70
	smaller_image_size = [100,100] # sized based on 'speed' of joint across screen
	smaller_frames = [None,None]
	# initialised for first iteration (when no previous position)
	y_tops = [0,0]
	y_bottoms = [1080,1080] #might need an adaptive way of doing this
	heights = [1080,1080]
	x_lefts = [0,0]
	x_rights = [1920,1920]
	widths = [1920,1920]

	previous_xs = [None,None]
	previous_ys = [None,None]
    
	# set window heights to be used, need to make it fast, adaptive and viable
	# start: fit in frame, end: not much smaller than viable joint image
	window_sizes = []
	ws = int(min(heights[0],widths[0])/9)
	for i in range(7):
		window_sizes.append(ws)
		ws = int(ws*0.75)
	#window_sizes = [100,80,60,50,45,40]
	#print(window_sizes)
	jd.set_window_sizes(window_sizes) # need to make these adaptive somehow, heatmap sizes?
    
	print('window frame sizes:',window_sizes)
	#input('wait')

	c = 0

	while(c<173):

		start = clock()
        
		try:
			_,frame = cap.read()
		except:
			break
        
		for model_num in range(len(models)):
			print('current model number:',model_num)
			for sf in numofjoints[model_num]:
				smaller_frames[sf] = frame[y_tops[sf]:y_bottoms[sf],x_lefts[sf]:x_rights[sf]]
				smaller_frames[sf] = jd.prepare_image_test(smaller_frames[sf])
			try:
				smaller_frames_to_process = []
				im_displacements = []
				for sf in numofjoints[model_num]:
					smaller_frames_to_process.append(smaller_frames[sf])
					im_displacements.append([x_lefts[sf],y_tops[sf]])
				hm,h_val = jd.get_heatmap(models[model_num],smaller_frames_to_process,im_displacements)
				locations[model_num] = jd.get_k_centroids(hm,h_val,len(numofjoints[model_num]))
			except Exception as e:
				print(e)
				for sf in numofjoints[model_num]:
					y_tops[sf] = 0
					y_bottoms[sf] = 1080 #might need an adaptive way of doing this
					heights[sf] = 1080
					x_lefts[sf] = 0
					x_rights[sf] = 1920
					widths[sf] = 1920
				smaller_frames_to_process = []
				im_displacements = []
				for sf in numofjoints[model_num]:
					smaller_frames_to_process.append(frame)
					im_displacements.append([x_lefts[sf],y_tops[sf]])
				hm,h_val = jd.get_heatmap(models[model_num],smaller_frames_to_process,im_displacements)
				locations[model_num] = jd.get_k_centroids(hm,h_val,len(numofjoints[model_num]))
				# maybe extrapolate from last position with velocity or something
			print('locations found:',locations)
			#try:
			#    for l in locations:
			#        l[0] = l[0]+x_left
			#        l[1] = l[1]+y_top
			#except:
			#    print('None type not int coords')
			#locations.append(ls)
			
			# try calculate relative speeds of joints moving across the screen
			#try:
			#    s = sqrt((locations[-1][0]-locations[-2][0])**2+(locations[-1][1]-locations[-2][1])**2)
			#    vx = (locations[-1][0]-locations[-2][0])/s
			#    vy = (locations[-1][1]-locations[-2][1])/s
			#except:
			#    s = 0
			#    vx = 0
			#    vy = 0
			
			#speeds.append(s)
			#velocities.append([vx,vy])
			
			#smaller_image_size = [base_size+int(s*1.5),base_size+int(s*1.5)]
			
			print(clock()-start)
			
			#find the joint near to where it was last found
			#for model_num in range(1):
			for jnum in numofjoints[model_num]:
				previous_xs[jnum] = locations[model_num][jnum][0]
				previous_ys[jnum] = locations[model_num][jnum][1]
				#print('model:',model_num,'joint:',jnum)
				# stop it frame grabbing a frame-segment outside the frame
				y_tops[jnum] = previous_ys[jnum]-smaller_image_size[0]
				y_bottoms[jnum] = previous_ys[jnum]+smaller_image_size[0]
				x_lefts[jnum] = previous_xs[jnum]-smaller_image_size[1]
				x_rights[jnum] = previous_xs[jnum]+smaller_image_size[1]
				if(y_tops[jnum]<0):
					y_tops[jnum] = 0
				if(y_bottoms[jnum]>heights[jnum]):
					y_bottoms[jnum] = heights[jnum]
				if(x_lefts[jnum]<0):
					x_lefts[jnum] = 0
				if(x_rights[jnum]>widths[jnum]):
					x_rights[jnum] = widths[jnum]

			if(c%5==0):
				print(locations[model_num])
				for l in locations[model_num]:
					cv2.circle(frame,(l[0],l[1]),3,[0,0,255],-1)
				
				#cv2.imwrite('video_frame.png',frame)
				#input('wait')

		# still should be larger than a single window size at least
		if(c%5==0):
			cv2.imshow('frame',frame)
			cv2.waitKey()
			cv2.destroyAllWindows()
		print(c)
		c += 1

	#for i in range(len(locations)):
	#    print(locations[i])
	#print(speeds[i])
	#print(velocities[i])

	return locations



locations = video_track()
input('press enter when ready')
#play_video_with_locations(locations)








































































































#############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################