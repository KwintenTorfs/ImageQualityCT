import numpy as np
from scipy import ndimage

import imageQualityCT.dicomProcessing as dcp
import matplotlib.pyplot as plt

path = r'D:\Database  - Summer Project RB\Database - ViewDEX study\LIDC-IDRI\LIDC-IDRI_Patient02\CT'
path2 = r'C:\Users\ktorfs5\KU Leuven\PhD\07 Students\04 Material for Students\01 Biomedical Interns 2023-2024\Exercises Viewer\Lungman Fantoom\Lungman - Dik - Reconstructie 2'


dicom = dcp.Image(path, '1-055.dcm')
# dicom = dcp.Image(path2, 'CT.1.3.12.2.1107.5.1.4.83812.30000023092609135974300031651')

image = dicom.raw_hu
dcm = dicom.dicom
minimal = np.min(image)

recon = dicom.ReconstructionTargetCenter
reconD = dicom.ReconstructionDiameter
dcc = dicom.DataCollectionCenter
dcd = dicom.DataCollectionDiameter
if recon is None or dcc is None:
    vector = [0, 0, 0]
else:
    vector = recon - dcc

px = dicom.PixelSize
distance_to_center = np.zeros(image.shape)
c1, c2 = (np.array(distance_to_center.shape) - 1) / 2
for x, row in enumerate(distance_to_center):
    for y, col in enumerate(row):
        r = np.sqrt((x - c1) ** 2 + (y - c2) ** 2)
        distance_to_center[x, y] = r * px

fov_map = np.zeros(image.shape)
v1, v2, _ = vector
for x, row in enumerate(fov_map):
    for y, col in enumerate(row):
        r = np.sqrt(((x - c1) * px - v1) ** 2 + ((y - c2) * px - v2) ** 2)
        fov_map[x, y] = r

mask = fov_map < dcd / 2
segment = mask.astype(np.float32)
segment[segment == 0] = np.nan
good = image * segment
plt.imshow(image)
plt.show()
plt.imshow(good, cmap='gray', vmin=-2100, vmax=1100)
plt.show()

print('Reconstruction Diameter = %.2f mm' % reconD)
print('Data Collection Diameter = %.2f mm' % dcd)
print('dFOV = %.2f mm' % (512 * px))

#
# mask = np.zeros(image.shape)
# mask[image == minimal] = 1
#
# plt.subplot(221)
# plt.imshow(mask)
# plt.title('Mask')
# plt.axis('off')
# plt.subplot(222)
# plt.imshow(image, cmap='gray', vmin=-1200, vmax=100)
# plt.title('Raw Image')
# plt.axis('off')
# plt.subplot(223)
# gauss = ndimage.gaussian_filter(mask, 3)
# plt.imshow(gauss)
# plt.title('Smooth Mask')
# plt.axis('off')
# plt.show()