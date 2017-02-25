# Attribution:
# pre-processig steps are largely based on the tutorial:
# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

from __future__ import print_function
import os
import numpy as np
import dicom
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Run Options
singleImage = 0



# Paths
imageDir='/home/vaibhavs/Projects/LungCancerDetection/data/stage1/'
patients=os.listdir(imageDir)
patients.sort()


# Other Parameters
# Following two parameters are tied with one another
# TODO: Zooming to result in to a given order

image_size=[100, 24, 24]

# Procedure for loading stack of scan images for one patient
def load_scan(path):
    slices=[dicom.read_file(path+'/'+s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

# procedure for converting the slice array into np array in Hounsfield Units
def get_pixels_hu(slices):

    image=np.stack([s.pixel_array for s in slices]).astype(np.int16)
    # Out of bound pixels for scanner
    image[image== -2000] = 0

    # Aparently in DiCOM format each slice also has attributes called
    # RescaleSlope and RescaleIntercept and conversion to HU can be done by
    #  h = slope*p + intercept
    # where p is pixel value, h is the correspoding  value in Hounsefield units
    # slope and intercepts are attributes of the slice to which the pixel
    # belongs

    for slice_number in xrange(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        image[slice_number] = slope * image[slice_number].astype(np.float64)
        image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


train_label_file='/home/vaibhavs/Projects/LungCancerDetection/data/stage1_labels.csv'
train_label_dict=pd.read_csv(train_label_file, index_col=0).to_dict()
total_train=len(train_label_dict['cancer'].keys())

test_label_file='/home/vaibhavs/Projects/LungCancerDetection/data/stage1_sample_submission.csv'
test_label_dict=pd.read_csv(test_label_file, index_col=0).to_dict()
total_test=len(test_label_dict['cancer'].keys())


allImages=os.listdir(imageDir)
num_classes=2

#Create a container to store all the image data
train_images=np.ndarray((total_train, image_size[0], image_size[1], image_size[2]),
        dtype=np.uint8)
test_images=np.ndarray((total_test, image_size[0], image_size[1], image_size[2]),
        dtype=np.uint8)
test_image_ids=[]


train_labels=np.ndarray((total_train, num_classes))
i=0
j=0


for imageset_id in allImages:

    print("Processing : %s ...." %imageset_id)
    imageset=load_scan(imageDir+imageset_id)
    imageset2pixels=get_pixels_hu(imageset)
    newImage  = scipy.ndimage.interpolation.zoom(imageset2pixels,
             np.array(map(float, image_size))/list(imageset2pixels.shape))

    if imageset_id in train_label_dict['cancer']:
        train_labels[i]= train_labels[i]*0
        train_images[i]=newImage
        train_labels[i][train_label_dict['cancer'][imageset_id]]=1
        i+=1
    else:
        test_image_ids.append(imageset_id)
        test_images[j]=newImage
        j+=1


test_ids=np.array(list(test_image_ids))



np.save('imgs_train_data.npy', train_images)
np.save('imgs_test_data.npy', test_images)
np.save('test_ids.npy', test_ids)
np.save('train_labels.npy', train_labels)
print("Total number of sets processed: %d" %len(allImages))
print('Saving to .npy files done!')




