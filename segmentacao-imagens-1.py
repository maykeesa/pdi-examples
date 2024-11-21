import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.color import label2rgb
from skimage.measure import regionprops

# Thresholding (limiarização)

img = data.page()

  #Todos os tipos de Thresholding
fig,ax = try_all_threshold(img, figsize=(10,8), verbose=False)
plt.show()

  # Thresholding de forma manual
    # Como descobrir o melhor binary?
binary = image > 40
plt.imshow(binary, cmap="gray")

  # Thresholding Otsu
thresh = threshold_otsu(image)
binaryTO = image > thresh

plt.imshow(binaryTO, cmap = "gray")

  # Thresholding local
block_size = 35
local_thresh = threshold_local(new_img, block_size, offset = 10)
binary_local = new_img > local_thresh

plt.imshow(binary_local, cmap="gray")

  # Thresholding com OpenCV
img = cv.imread("/content/img01.jpeg",0)
plt.imshow(img, cmap="gray")

  # Global
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

  # Local - Media
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                           cv.THRESH_BINARY,11,2)

   # Local - Soma Gaussiana Ponderada
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                           cv.THRESH_BINARY,11,2)

plt.imshow(th1, 'gray')
plt.imshow(th2, 'gray')
plt.imshow(th3, 'gray')

  # Etiquetar regiões da imagem
image = data.coins() [50:-50, 50:-50]
plt.imshow(image, cmap = 'gray')

thresh = threshold_otsu(image)
binaryC = image > thresh
plt.imshow(binaryC, cmap='gray')

newBinary = closing(binaryC, square(5))
plt.imshow(newBinary, cmap='gray')

imgCB = clear_border(newBinary)
plt.imshow(imgCB, cmap='gray')

li = label(imgCB)
plt.imshow(li, cmap='gray')

image_label_overlay = label2rgb(li, image = image)
plt.imshow(image_label_overlay, cmap='gray')

fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(image_label_overlay)

for region in regionprops(li):
  if region.area >= 100:
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill = False, edgecolor = 'red', linewidth = 2)
    ax.add_patch(rect)
