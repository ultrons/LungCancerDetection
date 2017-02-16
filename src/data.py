# Attribution:
# pre-processig steps are largely based on the tutorial:
# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

from __future__ import print_function
import os
import numpy as np
import dicom
import scipy
import matplotlib.pyplot as plt

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

image_size=[100, 64, 64]
rescaling_factor=[1, 4, 4] # Isomorphic resolution of 4mm, 4mm????

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


# resampling procedure takes the image(np array)
# scan is the list of slices, where each dicom slice
# pixel distance is the distance between centers of each two dimensional pixel

def resample(image, scan, new_spacing):
    #current pixel spacing
    spacing=map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing=np.array(list(spacing))


    resize_factor = spacing / new_spacing # elementwise division
    new_real_shape = image.shape * resize_factor # elementwise multiplication
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing





if singleImage == 1:
    sample_slices=load_scan(imageDir + '00cba091fa4ad62cc3200a657aeb957e')
    sample_slice_hu=get_pixels_hu(sample_slices)
    pix_resampled, spacing = resample(sample_slice_hu, sample_slices, [1,4,4])
    print("Shape before resampling\t", sample_slice_hu.shape)
    print("Shape after resampling\t", pix_resampled.shape)
    print(sample_slice_hu.shape)
    plt.imshow(sample_slice_hu[80], cmap=plt.cm.gray)
    plt.show()

    exit()

allImages=os.listdir(imageDir)
total=len(allImages)

#Create a container to store all the image data
images=np.ndarray((total, image_size[0], image_size[1], image_size[2]),
        dtype=np.uint8)
i=0
for imageset_id in allImages:
    print("Processing : %s ...." %imageset_id)
    imageset=load_scan(imageDir+imageset_id)
    imageset2pixels=get_pixels_hu(imageset)
    #newImage, spacing = resample(imageset2pixels, imageset, rescaling_factor)
    newImage  = scipy.ndimage.interpolation.zoom(imageset2pixels,
            np.array(list(imageset2pixels.shape))/image_size)
    try:
        images[i]=newImage
    except:
        print("Exception for Patient:%s" %imageset_id)
        print("Container Shape:", images[i].shape)
        print("Shape of patient imageset:", newImage.shape)

np.save('imgs_data.npy', images)
print("Total number of sets processed: %d" %total)
print('Saving to .npy files done!')
