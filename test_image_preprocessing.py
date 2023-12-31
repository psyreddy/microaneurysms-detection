# Imports

import cv2
import os
import random
import math
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot
from patchify import patchify, unpatchify

# Function for centering Microaneurysms

def find_center(contours):
    y = []
    for i in range(len(contours)):
        a = contours[i]
        center = np.average(a,axis=0)
        x_cen = math.ceil(center[0,0])
        y_cen = math.ceil(center[0,1])
        x = [x_cen,y_cen]
        y.append(x)
    return y

# Finding Elements in Test Data Set

x_test_path = 'dataSets/testDatasetEoptha/x_test'
y_test_path = 'dataSets/testDatasetEoptha/y_test'

x_test_list = os.listdir(x_test_path)
y_test_list = os.listdir(y_test_path)

if not os.path.isdir('dataSets/preprocessed/testpatches/MA/x'):
    os.makedirs('dataSets/preprocessed/testpatches/MA/x')
if not os.path.isdir('dataSets/preprocessed/testpatches/MA/y'):
    os.makedirs('dataSets/preprocessed/testpatches/MA/y')

# Preprocessing of x_test , y_test with MA at center of image

val = 64
e = (128,128)
for i in tqdm(range(len(x_test_list))):
    x_read = cv2.imread('dataSets/testDatasetEoptha/x_test/'+x_test_list[i],cv2.IMREAD_UNCHANGED)
    x_resized = cv2.resize(x_read,(2432,1664),interpolation=cv2.INTER_CUBIC)
    x_green = x_resized[:,:,1]
    x_histeq = cv2.equalizeHist(x_green)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x_final = clahe.apply(x_histeq)

    y_read = cv2.imread('dataSets/testDatasetEoptha/y_test/'+y_test_list[i],cv2.IMREAD_UNCHANGED)
    y_resized = cv2.resize(y_read, (2432,1664),interpolation=cv2.INTER_CUBIC)
    contours, h = cv2.findContours(image=y_resized, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    y = find_center(contours)
    for j in range(len(y)):
        if y[j][0]-val>0 and y[j][1]-val>0:
            x_cen_img = x_final[y[j][1]-val:y[j][1]+val,y[j][0]-val:y[j][0]+val]
            y_cen_img = y_resized[y[j][1]-val:y[j][1]+val,y[j][0]-val:y[j][0]+val]
            if e==y_cen_img.shape:
                cv2.imwrite('dataSets/preprocessed/testpatches/MA/x/'+'Cen_'+x_test_list[i][0:-4]+'_'+str(j)+'.png',x_cen_img)
                cv2.imwrite('dataSets/preprocessed/testpatches/MA/y/'+'Cen_'+x_test_list[i][0:-4]+'_'+str(j)+'.png',y_cen_img)
            else:
                pass
        else:
            pass

# Paths for pathches of x_test and y_test with NO MA

if not os.path.isdir('dataSets/preprocessed/testpatches/NonMA/x'):
    os.makedirs('dataSets/preprocessed/testpatches/NonMA/x')
if not os.path.isdir('dataSets/preprocessed/testpatches/NonMA/y'):
    os.makedirs('dataSets/preprocessed/testpatches/NonMA/y')

# Preprocessing of x_test , y_test with NO MA

for i in tqdm(range(len(x_test_list))):
    x_read = cv2.imread(x_test_path+'/'+x_test_list[i])
    x_resized = cv2.resize(x_read,(2432,1664),interpolation=cv2.INTER_CUBIC)
    x_green = x_resized[:,:,1]
    x_histeq = cv2.equalizeHist(x_green)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x_final = clahe.apply(x_histeq)

    y_read = cv2.imread(y_test_path+'/'+y_test_list[i],cv2.IMREAD_UNCHANGED)
    y_resized = cv2.resize(y_read,(2432,1664),interpolation=cv2.INTER_CUBIC)
    y_val,y_final = cv2.threshold(y_resized,127,255,cv2.THRESH_BINARY)

    x_patches1 = patchify(x_final,patch_size=(128,128),step=128)
    x_patches = np.squeeze(x_patches1)
    y_patches = patchify(y_final,patch_size=(128,128),step=128)

    for x_j,y_j in zip(range(x_patches.shape[0]),range(y_patches.shape[0])):
        for x_k,y_k in zip(range(x_patches.shape[1]),range(y_patches.shape[1])):
            x_patch_img = x_patches[x_j,x_k,:,:]
            y_patch_img = y_patches[y_j,y_k,:,:]

            y_unique = np.unique(np.reshape(y_patch_img,-1))
            if 255 in y_unique:
                pass
            else:
                cv2.imwrite('dataSets/preprocessed/testpatches/NonMA/x/'+x_test_list[i][:-4]+'_'+str(x_j)+'_'+str(x_k)+'.png',x_patch_img)
                cv2.imwrite('dataSets/preprocessed/testpatches/NonMA/y/'+y_test_list[i][:-4]+'_'+str(y_j)+'_'+str(y_k)+'.png',y_patch_img)