import numpy as np
import cv2
import time
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import time

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from scipy.ndimage.measurements import label

##############################################################################

##############################################################################
# Utils

def load_test_images(glob_regex='test_images/*.jpg'):
	images = []
	fnames = []
	f = []
	for f in glob.iglob(glob_regex, recursive=True):
		fnames.extend(glob.glob(f))
		# img = cv2.imread(f)
		# img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# images.append(img)
		# print(f,img.shape)
	# return images, fnames
	return fnames

def load_test_video(file_name='test_video.mp4'):
	vimages = []
	vframes = []
	count = 0
	clip = VideoFileClip(file_name)
	for img in clip.iter_frames(progress_bar=True):
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		vimages.append(img)
		vframes.append("%s - %d" % (file_name, count))
		count += 1

	return vimages, vframes

##############################################################################

##############################################################################
# Classifiers

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

def extract_hog_features(img, vis=False, feature_vec=False):
	"""
	Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
	visualization. Features will always be the first return (flattened if feature_vector= True).
	A visualization matrix will be the second return if visualize = True.
	"""
	global orient, pix_per_cell, cell_per_block
	hog_features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block),
								  block_norm= 'L2-Hys', transform_sqrt=False, 
								  visualize= True, feature_vector= feature_vec)
	if vis:
		return hog_features, hog_image
	else:
		return hog_features

def extract_per_frame(img):
	# Load parameters
	global spatial_feat, hist_feat, hog_feat
	global cspace, spatial_size, hist_bins, hist_range
	global orient, pix_per_cell, cell_per_block, hog_channel

	img_features = []
	# Apply color conversion if other than 'RGB'
	if cspace != 'RGB':
		if cspace == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif cspace == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif cspace == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif cspace == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif cspace == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(img)

	# Compute spatial features if flag is set
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		#4) Append features to list
		img_features.append(spatial_features)
	# Compute histogram features if flag is set
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
		#6) Append features to list
		img_features.append(hist_features)
	# Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(extract_hog_features(feature_image[:,:,channel], 
														vis=False, feature_vec=True))	  
		else:
			hog_features = extract_hog_features(feature_image[:,:,hog_channel],
												vis=False, feature_vec=True)
		# Append features to list
		img_features.append(hog_features)

	# Return concatenated array of features
	return np.concatenate(img_features)

def extract_features(imgs):
	features = []
	for file in imgs:
		# Read in each one by one
		img = mpimg.imread(file)
		features_per_frame = extract_per_frame(img)
		features.append(features_per_frame)
	return features

##############################################################################
#################################### Main ####################################
##############################################################################

timestamp_start = time.time()
# Color Feature Parameters:
cspace = 'YCrCb'
spatial_size = (32, 32)
hist_bins = 32
hist_range = (0, 256)
# HOG Feature Paramters:
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
# Feature enable
spatial_feat=True
hist_feat=True
hog_feat=True

##############################################################################
# Training data (comment out once done)

# # Load Images
# files_non = load_test_images('./non-vehicles/**/*.png')
# files_veh = load_test_images('./vehicles/**/*.png')
# print("Training images loaded")
# # Extract features
# features_non = extract_features(files_non)
# features_veh = extract_features(files_veh)

# # Save classified training data
# joblib.dump(features_non, 'features_non.joblib')
# joblib.dump(features_veh, 'features_veh.joblib')
# print("Training data saved")
##############################################################################
# Load classified training data
features_non = joblib.load('features_non.joblib')
features_veh = joblib.load('features_veh.joblib')
print("Training data loaded")
##############################################################################
# Predict test images

# Create an array stack of feature vectors
X = np.vstack((features_veh, features_non)).astype(np.float64)  
# Define the labels vector
y = np.hstack((np.ones(len(features_veh)), np.zeros(len(features_non))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)

# # Save X_scaler
# joblib.dump(X_scaler, 'X_scaler.joblib')
# Load X_scaler
X_scaler = joblib.load('X_scaler.joblib')

# # Apply the scaler to X
# X_train = X_scaler.transform(X_train)
# X_test = X_scaler.transform(X_test)

# # Use a linear SVC 
# svc = LinearSVC()
# # svc = DecisionTreeClassifier(criterion='entropy',min_samples_split=30,random_state=42)

# # Check the training time for the SVC
# t = time.time()
# svc.fit(X_train, y_train)
# t2 = time.time()
# print(round(t2-t, 2), 'Seconds to train SVC...')

# # Save trained model
# joblib.dump(svc, 'svc.joblib')
# Load trained model
svc = joblib.load('svc.joblib')

# # Check the score of the SVC
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# # Check the prediction time for a single sample
# t=time.time()
# n_predict = 10
# print('My SVC predicts:\t', svc.predict(X_test[0:n_predict]))
# print('For these\t',n_predict, 'labels: ', y_test[0:n_predict])
# t2 = time.time()
# print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# timestamp_finish = time.time()
# print('Total time: ',round((timestamp_finish-timestamp_start), 5), 'seconds')

##############################################################################
##############################################################################
##############################################################################


##############################################################################
# Sliding window / hot box technique

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = int(img.shape[0]/2)
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	# Compute the span of the region to be searched	
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows):
	# Load trained model
	svc = joblib.load('svc.joblib')
	# Load X_scaler
	X_scaler = joblib.load('X_scaler.joblib')
	#1) Create an empty list to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for window in windows:
		# print(window)
		#3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))	  
		#4) Extract features for that window using single_img_features()
		features = extract_per_frame(test_img)
		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
			print("Found")
	#8) Return windows for positive detections
	return on_windows

##############################################################################
##############################################################################
# Test on images

# fnames = []
# for i in range(1,7):
# 	fname = './test_images/test' + str(i) + '.jpg'
# 	fnames.extend(glob.glob(fname))
# 	#fnames = glob.glob(fname)

# for i, fname in enumerate(fnames):
# 	print(fname)
# 	img = mpimg.imread(fname)
# 	img = img.astype(np.float32)/255			# convert jpg format img to png format img

# 	windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
# 						xy_window=(128, 128), xy_overlap=(0.5, 0.5))
# 	hot_windows = search_windows(img, windows, svc, X_scaler)
# 	window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)

# 	fname_out = fname[:-4] + '_out.jpg'
# 	# cv2.imwrite(fname_out, window_img)
# 	mpimg.imsave(fname_out, window_img)

# 	timestamp_finish = time.time()
# 	print('Time per frame: ',round((timestamp_finish-timestamp_start), 5), 'seconds')

# timestamp_finish = time.time()
# print('Total time: ',round((timestamp_finish-timestamp_start), 5), 'seconds')

##############################################################################

def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap# Iterate through list of bboxes
	
def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
	# Return the image
	return img

# Final pipeline
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img):
	# Define parameters
	ystart = 400
	ystop = 656
	scale = 1.25
	# Load trained model
	svc = joblib.load('svc.joblib')
	# Load X_scaler
	X_scaler = joblib.load('X_scaler.joblib')
	# Load parameters
	global orient, pix_per_cell, cell_per_block, spatial_size, hist_bins

	draw_img = np.copy(img)
	# convert jpg format img to png format img
	img = img.astype(np.float32)/255
	
	img_tosearch = img[ystart:ystop,:,:]
	fname_out = 'tmp.jpg'
	mpimg.imsave(fname_out, img_tosearch)
	ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
	nfeat_per_block = orient*cell_per_block**2
	
	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
	
	# Compute individual channel HOG features for the entire image
	hog1 = extract_hog_features(ch1)
	hog2 = extract_hog_features(ch2)
	hog3 = extract_hog_features(ch3)
	
	box_list = []
	heat = np.zeros_like(img[:,:,0]).astype(np.float)

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
		  
			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))	
			#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))	
			test_prediction = svc.predict(test_features)
			
			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				box = (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)
				box_list.append(box)
				# cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

	# Add heat to each box in box list
	heat = add_heat(heat,box_list)
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,1)
	# Visualize the heatmap when displaying	
	heatmap = np.clip(heat, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(draw_img, labels)

	return draw_img

# for i, fname in enumerate(fnames):
# 	print(fname)
# 	img = mpimg.imread(fname)

# 	out_img = find_cars(img)
# 	fname_out = fname[:-4] + '_out.jpg'
# 	# cv2.imwrite(fname_out, window_img)
# 	mpimg.imsave(fname_out, out_img)

##############################################################################
# Run vieo through pipeline

from moviepy.editor import VideoFileClip

video_input = 'project_video.mp4'
video_output = video_input[:-4] + '_out.mp4'
clip = VideoFileClip(video_input)
output_clip = clip.fl_image(find_cars)
output_clip.write_videofile(video_output, audio=False)