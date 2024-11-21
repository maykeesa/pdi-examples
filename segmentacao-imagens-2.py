import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.util import img_as_ubyte
from skimage.morphology import erosion, dilation
from skimage.morphology import disk

# Metodo de Contorno Ativo
img = data.astronaut()
img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 400)
r = 100 + 100*np.sin(s)
c = 220 + 100*np.cos(s)

init = np.array([r, c]).T
snake = active_contour(gaussian(img, 3, preserve_range=False),
                       init, alpha=0.015, beta=10, gamma=0.001)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)

ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)

ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()

# Métodos morfológicos

  # Erosão
def plot_comparacao(original, filtrada, nome):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('imagem original')
    ax1.axis('off')
    ax2.imshow(filtrada, cmap=plt.cm.gray)
    ax2.set_title(nome)
    ax2.axis('off')

imgErosao = img_as_ubyte(data.shepp_logan_phantom())
fig, ax = plt.subplots()
ax.imshow(imgErosao, cmap=plt.cm.gray)

elementoEstruturante = disk(3)
erosao = erosion(imgErosao,elementoEstruturante)
plot_comparacao(imgErosao, erosao, 'imagem - erosão')

elementoEstruturante = disk(6)
erosao = erosion(imgErosao,elementoEstruturante)
plot_comparacao(imgErosao, erosao, 'imagem - erosão')

  # Diltação
elementoEstruturante = disk(6)
dilatacao = dilation(imgErosao,elementoEstruturante)

plot_comparacao(imgErosao, dilatacao, 'imagem - dilatação')

  # Erosão e Dilatação com OpenCV
img = cv.imread("/content/estacionamento.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(8, 10))
plt.imshow(img)

#----------------------------------

imagem_gray = cv.imread("/content/estacionamento.jpg", cv.IMREAD_GRAYSCALE)
plt.figure(figsize=(8, 10))
plt.imshow(imagem_gray, cmap="gray")

#----------------------------------

imagem_suavizada = cv.GaussianBlur(imagem_gray, (1,1), 0)
ret, imagem_limiarizada = cv.threshold(imagem_suavizada, 180, 255, cv.THRESH_BINARY)
plt.figure(figsize=(8, 10))
plt.imshow(imagem_limiarizada, cmap="gray")

#----------------------------------

kernel = np.ones((1,1), np.uint8)
imagem_erosao = cv.erode(imagem_limiarizada, kernel, iterations=1)
plt.figure(figsize=(8, 10))
plt.imshow(imagem_erosao, cmap="gray")

#----------------------------------

kernel = np.ones((3,3), np.uint8)
imagem_dilatacao = cv.dilate(imagem_limiarizada, kernel, iterations=2)
plt.figure(figsize=(8, 10))
plt.imshow(imagem_dilatacao, cmap="gray")

  # Identificação de objetos e formas
imgMulherGray = cv.imread("/content/mulher.jpg", cv.IMREAD_GRAYSCALE)
plt.figure(figsize=(20, 10))
plt.imshow(imgMulherGray, cmap="gray")

bordaCanny = cv.Canny(imgMulherGray, 40, 80)
plt.figure(figsize=(20, 10))
plt.imshow(bordaCanny, cmap="gray")

_, mask = cv.threshold(bordaCanny, 30, 255, cv.THRESH_BINARY_INV)
plt.figure(figsize=(20, 10))
plt.imshow(mask, cmap="gray")

#----------------------------------

imgObjetos = cv.imread("/content/objetos.jpeg")
imgObjetos = cv.cvtColor(imgObjetos, cv.COLOR_BGR2RGB)
plt.figure(figsize=(20, 20))
plt.imshow(imgObjetos)

imgObjetosGray = cv.cvtColor(imgObjetos,cv.COLOR_BGR2GRAY)
imgBordas = cv.Canny(imgObjetosGray, 1, 100)
plt.figure(figsize=(20, 20))
plt.imshow(imgBordas)

contornosObjetos, hierarchy_objetos = cv.findContours(imgBordas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print("Contornos encontrados: "+str(len(contornosObjetos)))

imgObjetosContornos = imgObjetosGray.copy()
cv.drawContours(imgObjetosContornos, contornosObjetos, -1, (0,255,0), 3)
plt.figure(figsize=(20, 20))
plt.imshow(imgObjetosContornos)

#----------------------------------

imgFormas = cv.imread("/content/formas.png")
imgFormas = cv.cvtColor(imgFormas, cv.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(imgFormas)

imgFormasGray = cv.cvtColor(imgFormas, cv.COLOR_BGR2GRAY)
imgFormasGraySuavizada = cv.GaussianBlur(imgFormasGray, (3,3), 0)
imgFormasBordas = cv.Canny(imgFormasGraySuavizada, 40, 180)

plt.figure(figsize=(20, 10))
plt.imshow(imgFormasBordas)

contornosFormas, hierarchy = cv.findContours(imgFormasBordas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print("Contornos encontrados "+str(len(contornosFormas)))

imagemFormasContornosInternos = imgFormas.copy()

cv.drawContours(imagemFormasContornosInternos, contornosFormas, -1, (0,255,0), 2)

plt.figure(figsize=(20, 10))
plt.imshow(imagemFormasContornosInternos)
