# Imports

import cv2
import os
import random
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

# Finding Elements in Train Data Set

x_train_path = 'dataSets/trainDatasetEoptha/x_train'  # path to x_train
y_train_path = 'dataSets/trainDatasetEoptha/y_train'  # path to y_train

x_train_list = os.listdir(x_train_path)  # elements in x_train
y_train_list = os.listdir(y_train_path)  # elements in y_train

if not os.path.isdir('dataSets/preprocessed/trainpatches/MA/x'):
    os.makedirs('dataSets/preprocessed/trainpatches/MA/x')
if not os.path.isdir('dataSets/preprocessed/trainpatches/MA/y'):
    os.makedirs('dataSets/preprocessed/trainpatches/MA/y')

# Function for centering Microaneurysms

def find_center(contours):
    y = []
    for i in range(len(contours)):
        a = contours[i]
        center = np.average(a, axis=0)
        x_cen = math.ceil(center[0, 0])
        y_cen = math.ceil(center[0, 1])
        x = [x_cen, y_cen]
        y.append(x)
    return y

# Preprocessing of x_train , y_train with MA at center of image

val = 64
e = (128,128)

for i in tqdm(range(len(x_train_list))):
    x_read = cv2.imread('dataSets/trainDatasetEoptha/x_train/'+x_train_list[i])
    x_resized = cv2.resize(x_read, (2432,1664),interpolation=cv2.INTER_CUBIC)
    x_green = x_resized[:,:,1]
    x_histeq = cv2.equalizeHist(x_green)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x_final = clahe.apply(x_histeq)

    y_read = cv2.imread('dataSets/trainDatasetEoptha/y_train/'+y_train_list[i],cv2.IMREAD_UNCHANGED)
    y_resized = cv2.resize(y_read,(2432,1664),interpolation=cv2.INTER_CUBIC)
    y_val,y_thresold_applied = cv2.threshold(y_resized,127,255,cv2.THRESH_BINARY)
    contors,hier = cv2.findContours(image=y_thresold_applied,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    y = find_center(contors)
    for j in range(len(y)):
        if y[j][0]-val>0 and y[j][1]-val>0:
            x_cen_img = x_final[y[j][1]-val:y[j][1]+val,y[j][0]-val:y[j][0]+val]
            y_cen_img = y_resized[y[j][1]-val:y[j][1]+val,y[j][0]-val:y[j][0]+val]
            if e == y_cen_img.shape:
                x_patches_path = 'dataSets/preprocessed/trainpatches/MA/x/'+ 'cen_' + (x_train_list[i])[0:-4]+'_'+str(j)+'.png'
                y_patches_path = 'dataSets/preprocessed/trainpatches/MA/y/'+ 'cen_' + (x_train_list[i])[0:-4]+'_'+str(j)+'.png'
                cv2.imwrite(x_patches_path,x_cen_img)
                cv2.imwrite(y_patches_path,y_cen_img)
            else:
                pass
        else:
            pass

# Paths for pathches of x_train and y_train with NO MA

if not os.path.isdir('dataSets/preprocessed/trainpatches/NonMA/x'):
    os.makedirs('dataSets/preprocessed/trainpatches/NonMA/x')
if not os.path.isdir('dataSets/preprocessed/trainpatches/NonMA/y'):
    os.makedirs('dataSets/preprocessed/trainpatches/NonMA/y')

# Preprocessing of x_train , y_train with NO MA

for i in tqdm(range(len(x_train_list))):
    x_read = cv2.imread('dataSets/trainDatasetEoptha/x_train/'+x_train_list[i])
    x_resized = cv2.resize(x_read,(2432,1664),interpolation=cv2.INTER_CUBIC)
    x_green = x_resized[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x_final = clahe.apply(x_green)

    y_read = cv2.imread('dataSets/trainDatasetEoptha/y_train/'+y_train_list[i],cv2.IMREAD_UNCHANGED)
    y_resized = cv2.resize(y_read,  (2342,1664),interpolation=cv2.INTER_CUBIC)
    y_val,y_final = cv2.threshold(y_resized,127,255,cv2.THRESH_BINARY)

    x_patches = patchify(x_final,patch_size=(128,128),step=128)
    y_patches = patchify(y_final,patch_size=(128,128),step=128)

    for x_j,y_j in zip(range(x_patches.shape[0]),range(y_patches.shape[0])):
        for x_k,y_k in zip(range(x_patches.shape[1]),range(y_patches.shape[1])):
            x_patch_img = x_patches[x_j,x_k,:,:]
            y_patch_img = y_patches[y_j,y_k,:,:]

            y_unique = np.unique(np.reshape(y_patch_img,-1))
            if 255 in y_unique:
                pass
            else:
                cv2.imwrite('dataSets/preprocessed/trainpatches/NonMA/x/'+x_train_list[i][:-4]+'_'+str(x_j)+'_'+str(x_k)+'.png',x_patch_img)
                cv2.imwrite('dataSets/preprocessed/trainpatches/NonMA/y/'+y_train_list[i][:-4]+'_'+str(y_j)+'_'+str(y_k)+'.png',y_patch_img)