
"""
Coded by Hyunah
Date : 2019.05.16
V2 2019.08.02
"""

import os
import sys 
import pandas as pd
import glob
from sklearn.model_selection import GroupKFold,KFold
import random
import shutil



PATH0 = "/data/PNS/H_data/Preprocessing/3scale_enhancement/Enhancement_PNG/0_N"
# Normal data folder path
PATH1 = "/data/PNS/H_data/Preprocessing/3scale_enhancement/Enhancement_PNG/1_AbN"
# Abnormal data folder path


# Cropping ROI ranging from 350 to 950
def Cropping(imagepath):
	images = []

	for imagePath in glob.glob(path+'*.jpg'):
 	
		image = cv2.imread(imagePath)
		images.append(image)

		for i in range(len(images)):
			(img_h, img_w) = images[i].shape[:2]
			roi = images[i][350:950, :]	
	
	return roi

# Grouping patient for preventing the same patient image from being split into multiple folders 
def Grouping(path0, path1):

	file_list=pd.DataFrame(data = os.listdir(path0),columns=['name'],dtype=str)
	file_list1=pd.DataFrame(data = os.listdir(path1),columns=['name'],dtype=str)
	filelist = pd.merge(file_list,file_list1,how="outer")
	random.shuffle(filelist["name"])

	ID= []
	for i in range(len(filelist)):
		number = filelist["name"][i][:6]
		ID.append(number)	
	
	filelist["id"] = ID
	filelist["group"] = filelist.groupby(["id"]).ngroup()
	#FILE = filelist.groupby(["group"]).size().reset_index(name='count')

	group = filelist.drop_duplicates(["group"],keep = "first")
	
	return filelist

# Making 10 folders (n_fold) and remaing list
def Devision_to_n(p0, p1, n_fold = 10):
	
	n = (Grouping(p0, p1)).drop_duplicates(["group"],keep = "first")
	N = (pd.DataFrame(Grouping(p0, p1))).reset_index()
	G = pd.DataFrame({"group" : n["group"]})

	l = int(len(n)/n_fold)
	folder = pd.DataFrame()
	
	
	for i in range(n_fold):
		f =[]
		f.extend(G["group"][i*l:(i+1)*l])
		folder["f"+str(i)] = f

	last = G[l*10:]

	if os.path.isfile("/data/PNS/PNS_Classification/PNS/data/last.csv"):
		os.remove("/data/PNS/PNS_Classification/PNS/data/last.csv")
	last.to_csv("/data/PNS/PNS_Classification/PNS/data/last.csv")
		

	return folder

filelist = Grouping(PATH0, PATH1)
filelist.to_csv("/data/PNS/PNS_Classification/PNS/data/filelist.csv")
grouplist = Devision_to_n(PATH0,PATH1)
grouplist.to_csv("/data/PNS/PNS_Classification/PNS/data/grouplist.csv")


# Coping image files to n_fold for training and test
def Copy_file_to_nfold(PATH0, PATH1):

	filelist = Grouping(PATH0, PATH1)
	grouplist = Devision_to_n(PATH0,PATH1)	

	for i in range(10):
		to_path = "/data/PNS/PNS_Classification/PNS/data/10_fold_png/fold_"+str(i)+"/" 	
	
		d= pd.DataFrame({"name":filelist[filelist.group.isin(grouplist["f"+str(i)])]["name"]},dtype=str)
		D = d.reset_index()	
		D["PATH"] = ""
		D["dst"] = ""
		for j in range(len(D)):
			if D["name"][j][-9]== "0":
				D["PATH"][j] = PATH0
				D["dst"][j] = to_path + "0/"  
			else:			
				D["PATH"][j] = PATH1
				D["dst"][j] = to_path + "1/"
	
			src = D["PATH"][j]+"/"+D["name"][j]
			dst = D["dst"][j] + D["name"][j]
		
			shutil.copy2(src,dst)
	
	return 0

	#D.to_csv("d"+str(i)+".csv")
	#for i in range(len(D)):	
	#	src = D["PATH"][i]+"/"+D["name"][i]
	#	dst = D["dst"][i] + D["name"][i]
	
	#	shutil.copy2(src,dst)

			
		






	
	




