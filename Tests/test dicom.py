import numpy as np
from scipy import ndimage

import imageQualityCT.dicomProcessing as dcp
import matplotlib.pyplot as plt

path = r'C:\Users\ktorfs5\KU Leuven\PhD\09 Other\04 Quick Images\Test 5'


dicom = dcp.Image(path, '1-055.dcm')


image = dicom.array
dcm = dicom.dicom
minimal = np.min(image)
minimum = np.ones(image.shape)
minimum[image != 0] = 0
labels, label_nb = ndimage.label(minimum)
row, col = image.shape
corners = {'LU': (0, 0),
           'RU': (0, col - 1),
           'LD': (row - 1, 0),
           'RD': (row - 1, col - 1)}

labels_to_check = set()
for corner in corners.values():
    corner_label = labels[corner]
    if corner_label != 0:
        labels_to_check.add(corner_label)

plt.imshow(labels, vmin=0, vmax=5)
plt.show()
new_image = image.copy()
for label in labels_to_check:
    binary = np.zeros(labels.shape)
    binary[labels == label] = 1
    threshold_binary = np.array(ndimage.binary_fill_holes(binary))
    threshold_image = image.copy()
    threshold_image[~threshold_binary] = np.nan
    if np.nanmin(threshold_image) == np.nanmax(threshold_image):
        new_image[threshold_binary] = np.nan
    plt.imshow(binary)
    plt.show()

# plt.imshow(minimum, cmap='gray')
# plt.show()

plt.imshow(new_image, cmap='gray')
plt.show()
