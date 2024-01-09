import numpy as np
from scipy import ndimage

import imageQualityCT.dicomProcessing as dcp
import matplotlib.pyplot as plt


def circle_from_three_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    ss1 = x1 ** 2 + y1 ** 2
    ss2 = x2 ** 2 + y2 ** 2
    ss3 = x3 ** 2 + y3 ** 2
    a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    b = ss1 * (y3 - y2) + ss2 * (y1 - y3) + ss3 * (y2 - y1)
    c = ss1 * (x2 - x3) + ss2 * (x3 - x1) + ss3 * (x1 - x2)
    d = ss1 * (x3 * y2 - x2 * y3) + ss2 * (x1 * y3 - x3 * y1) + ss3 * (x2 * y1 - x1 * y2)
    x = - b / (2 * a)
    y = - c / (2 * a)
    r = np.sqrt((b ** 2 + c ** 2 - 4 * a * d) / (4 * a ** 2))
    return (x, y), r


def radial_distance_map(shape, center, px=1):
    radial_distance = np.zeros(shape)
    c1, c2 = center
    for x, row in enumerate(radial_distance):
        for y, col in enumerate(row):
            radial_distance[x, y] = px * np.sqrt((x - c1) ** 2 + (y - c2) ** 2)
    return radial_distance



path = r'C:\Users\ktorfs5\KU Leuven\PhD\09 Other\04 Quick Images\Test 5'
path2 = r'C:\Users\ktorfs5\KU Leuven\PhD\07 Students\04 Material for Students\01 Biomedical Interns ' \
        r'2023-2024\Exercises Viewer\Lungman Fantoom\Lungman - Dik - Reconstructie 2'


dicom = dcp.Image(path, '1-055.dcm')
dicom = dcp.Image(path2, 'CT.1.3.12.2.1107.5.1.4.83812.30000023092609135974300031615')


image = dicom.raw_hu
dcm = dicom.dicom
minimal = np.min(image)


mask = np.zeros(image.shape)
mask[image == minimal] = 1

nb = mask.shape[0]

distance_to_center = radial_distance_map(image.shape, (nb - 1) / 2)

distance1 = np.zeros(nb)
distance2 = np.zeros(nb)
diagonal1 = np.zeros(nb)
diagonal2 = np.zeros(nb)
for i in range(nb):
    diagonal1[i] = image[i, i]
    diagonal2[i] = image[nb - i - 1, i]
    distance1[i] = distance_to_center[i, i]
    distance2[i] = distance_to_center[nb - i - 1, i]

radii = np.ones(4) * np.infty
positions = []
if diagonal1[0] == minimal:
    for i, value in enumerate(diagonal1):
        if value != minimal:
            radii[0] = distance1[i - 1]
            positions.append((i - 0.5, i - 0.5))
            break

if diagonal1[-1] == minimal:
    for i, value in reversed(list(enumerate(diagonal1))):
        if value != minimal:
            radii[1] = distance1[i + 1]
            positions.append((i + 0.5, i + 0.5))
            break
if diagonal2[0] == minimal:
    for i, value in enumerate(diagonal2):
        if value != minimal:
            radii[2] = distance2[i - 1]
            positions.append(((nb - 0.5) - i, i - 0.5))
            break
if diagonal2[-1] == minimal:
    for i, value in reversed(list(enumerate(diagonal2))):
        if value != minimal:
            radii[3] = distance2[i + 1]
            positions.append(((nb - 1.5) - i, i + 0.5))
            break

centers = []
radii2 = []
for i, value in enumerate(positions):
    posis = positions.copy()
    posis.remove(value)
    center, r = circle_from_three_points(*posis)
    centers.append(center)
    radii2.append(r)

average_radius = np.average(radii2)
c1, c2 = (np.average([p[0] for p in centers]), np.average([p[1] for p in centers]))
circle_mask = np.zeros(image.shape)
for x, row in enumerate(circle_mask):
    for y, col in enumerate(row):
        r = np.sqrt((x - c1) ** 2 + (y - c2) ** 2)
        circle_mask[x, y] = r > average_radius


radius = np.min(radii)
print('Radius = %.2f' % radius)
mask2 = np.zeros(image.shape)
mask2[distance_to_center > radius] = 1
segment = mask2.astype(np.float32)
segment[segment == 0] = np.nan
good = image * segment
plt.subplot(221)
plt.imshow(circle_mask)
plt.title('Mask')
plt.axis('off')
plt.subplot(222)
plt.imshow(image, cmap='gray', vmin=-1200, vmax=100)
plt.title('Raw Image')
plt.axis('off')
plt.subplot(223)
plt.imshow(mask - mask2, cmap='gray', vmin=0, vmax=1)
plt.title('Mask based on diagonal')
plt.axis('off')
plt.subplot(224)
plt.imshow(circle_mask - mask, cmap='gray', vmin=0, vmax=1)
plt.title('Truncated')
plt.axis('off')
plt.show()