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



path = r'C:\Users\ktorfs5\KU Leuven\PhD\09 Other\04 Quick Images\Test 10'
path2 = r'C:\Users\ktorfs5\KU Leuven\PhD\07 Students\04 Material for Students\01 Biomedical Interns ' \
        r'2023-2024\Exercises Viewer\Lungman Fantoom\Lungman - Dik - Reconstructie 2'


dicom = dcp.Image(path, '1-055.dcm')
# dicom = dcp.Image(path2, 'CT.1.3.12.2.1107.5.1.4.83812.30000023092609135974300031615')


image = dicom.raw_hu
dcm = dicom.dicom
minimal = np.min(image)


minimum_mask = np.ones(image.shape)
minimum_mask[image == minimal] = np.nan

nb = image.shape[0]
center = ((nb - 1) / 2, (nb - 1) / 2)
distance_to_center = radial_distance_map(image.shape, center)

masked_distance = minimum_mask * distance_to_center

half = int(nb//2)
q1 = masked_distance[0:half, 0:half]
q2 = masked_distance[0:half, half:]
q3 = masked_distance[half:, 0:half]
q4 = masked_distance[half:, half:]

plt.subplot(221)
plt.imshow(q1)
plt.axis('off')
plt.subplot(222)
plt.imshow(q2)
plt.axis('off')
plt.subplot(223)
plt.imshow(q3)
plt.axis('off')
plt.subplot(224)
plt.imshow(q4)
plt.axis('off')
plt.tight_layout()
plt.show()