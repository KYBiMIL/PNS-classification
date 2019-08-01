import os, sys
import numpy as np
import random
import csv
import shutil
from skimage import io
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import cv2




'''
Coded by Juhui Lee
Date : 2019.5.16
Revised on 05.22

'''


def Make_Directory(path):
    ## Make a folder if not existing
    if not os.path.exists(path):
        os.makedirs(path)

    return True


def Crop_ROI(img, RL, upb = 350, lwb = 950):
    ## crop region of interest (ROI) ranging in 350 and 950 from the top
	if RL == 'R':
		cr_img = img[350:950, -400:]
	else:		# 'L'
		cr_img = image[350:950, :400]

	return cr_img		# 600x400



def Normalization(img):
    ## Normalize all pixels from an image by subtracting its mean and deviding from the standard devidation
	i_mean = img.mean()
	i_std = img.std()

	# normalized_img = (img - i_mean) / i_std

	epsilon = 1e-6
	normalized_img = (img - i_mean) / (i_std + epsilon)


	return normalized_img



def Enhancement(img, kernel_size = 19, gain = 1.):
    ## Enhance edges of an image
	kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
	smoothing = cv2.filter2D(img, -1, kernel)

	edge = (img - smoothing)
	enhanced_img = img + (edge * gain)

	E_offset = enhanced_img + abs( np.min(enhanced_img) )		# [0, ~)
	E_img = E_offset / np.E_offset * 255.							# [0, 255]

	return E_img




def CopyFiles(PATH_FROM, PATH_TO, NUM_F = 10, TEST_FOLD_IDX = '0'):
    ## Devide 'NUM_F'-folds to training and test for cross-validation

	#PATH_FROM = "/data/PNS/PNS_Classification/PNS/data/10_fold_juhui"
	#PATH_FROM = "/data/PNS/PNS_Classification/PNS/data/10_fold_png"

	#PATH_TO = "10_fold_test"
	#PATH_TO = "10_fold_cnn"

	#NUM_F = 10
	#TEST_FOLD_IDX = '0'


	folder_n = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(PATH_FROM) for f in files if all(s in f for s in ['.png'])]

	for i, f in enumerate(folder_n):
		
		temp_dir, temp_base = os.path.split(f)				# /.../10_fold_juhui/fold_n/0 or 1, ~.npy
		temp_fold, temp_class = os.path.split(temp_dir)	# /.../10_fold_juhui/fold_n, 0 or 1
		my_dir, fold_name = os.path.split(temp_fold)		# /.../10_fold_juhui, fold_n
		rt, ch_dir = os.path.split(my_dir)					# /..., 10_fold_juhui

		if fold_name[-1] == TEST_FOLD_IDX:
			path_to = os.path.join(rt, PATH_TO, "test", temp_class, temp_base)
			# /.../10_fold_test/test/0 or 1/~.npy
		else:
			path_to = os.path.join(rt, PATH_TO, "train", temp_class, temp_base)
			# /.../10_fold_test/train/ 0 or 1/~.npy

		shutil.copy2(f, path_to)




def Copy_Files_from_List(f_list, l_list, path = "/data/PNS/H_data/Preprocessing/Enhancement", save_path = "/data/PNS/PNS_Classification/PNS/data/10_fold_juhui", num_f = 10):
    ## Devide all data into 'num_f'+1 folders
	filedir = os.path.join(path, "*")

	All_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in files if all(s in f for s in ['.npy'])]
	
	

	f_n = np.shape(f_list)[1]			# 594
	l_n = len(l_list)						# 6

	for i, p in enumerate(All_list):			# /..../~.npy for all files
		p_dir, p_base = os.path.split(p)				# /.../, ~.npy
		_, p_class = os.path.split(p_dir)				# 0 or 1
		p_name, ext = os.path.splitext(p_base)	# ~, .npy
		p_name = p_name[:6]					
		
		for k in range(num_f):				# for 10 times
		
			for j, q in enumerate(f_list[k]):	# ID of k-th folder
				if p_name == q:				# the same name
					path_from = p				# that file path
					path_to = os.path.join(save_path, "fold_" + str(k), p_class[0], p_base)	# with the same name in different directory
					shutil.copy2(path_from, path_to)	# copy from to

		for a, lasts in enumerate(l_list):			# for last times
			if p_name == lasts:
				path_from = p
				path_to = os.path.join(save_path, "last", p_class[0], p_base)
				shutil.copy2(path_from, path_to)

	

	return 0




def Listing(path = "/data/PNS/H_data/Preprocessing/Enhancement"):
    ## Make a list of all files from input, which is a directory, not to include duplicates
	patient_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in files if all(s in f for s in ['.npy'])]

	for i, p in enumerate(patient_list):
		filename = os.path.basename(p)
		fn, ext = os.path.splitext(filename)
		patient_list[i] = fn[0:6]

	patient_list = list(set(patient_list))

	return patient_list




def Shuffle_List(p_list, n = 3):
    ## Randomly shuffle the input list 'n'-times
	for i in range (3):	# 0, 1, 2
		random.shuffle(p_list)

	return p_list




def Division_to_N_Fold(p_list, num_f = 10):
    ## Make two lists for a 'num_f'-fold data list and a remaining list
	l = len(p_list)	
	n = int(l / num_f)

	n_list = []
	last_list = []

	for i in range(num_f):	
		n_list += [p_list[n*i : n*(i+1)]]

	last_list = p_list[num_f*n:]


	if os.path.isfile("last_list_juhui.txt"):
		os.remove("last_list_juhui.txt")

	with open('last_list_juhui.txt', 'w') as f:
		for p_name in last_list:
			f.write("%s\n" % p_name)
		f.close()

	return n_list, last_list




def NPY2PNG(PATH):
    ## Transform ~.npy to ~.png extention
		
	# PATH = "/data/PNS/PNS_Classification/PNS/data/10_fold_test"
	# PATH = "/data/PNS/PNS_Classification/PNS/data/npti"


	folder_n = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(PATH) for f in files if all(s in f for s in ['.npy'])]


	for i, p in enumerate(folder_n):

		f_dir, f_name = os.path.split(p)
		f_, ext = os.path.splitext(f_name)
		png_path = os.path.join(f_dir, f_)

		img_arr = np.load(p)

		
		img255 = img_arr + abs(np.min(img_arr))
		img255 = img255 / np.max(img255) * 255.
		
		cv2.imwrite(png_path + ".png", img255)





