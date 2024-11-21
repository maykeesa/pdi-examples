import numpy as np
import cv2
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

# Filtragem
  # Filtros Lineares

    # Passa-baixa - Média:
image = cv2.imread('/content/img13.png')

figure_size = 9
new_image = cv2.blur(image,(figure_size, figure_size))
plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(image)
plt.title('Original')

plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(new_image)
plt.title('Passa-baixa: média')
plt.xticks([])
plt.yticks([])
plt.show()

  # Filtros Não-Lineares

    # Passa-baixa - Mediana:
image = cv2.imread('/content/img13.png')

figure_size = 9
median = cv2.medianBlur(image, figure_size)

plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(image)
plt.title('Original')

plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(median)
plt.title('Passa-baixa: mediana')
plt.xticks([])
plt.yticks([])
plt.show()

    # Passa-alta - Sovel
image = cv2.imread('/content/img13.png')

sobelx = cv2.Sobel(image,cv2.CV_64F,1,0)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1)
sobelxy = cv2.Sobel(image,cv2.CV_64F,1,1)

print("Imagem Original")
cv2_imshow(image)

print()

print("Passa-alta: Sobel")
cv2_imshow(sobelxy)

  # Filtragem no Domínimo Espacial
    
    # Correlação
def correlacao(img, mascara):
  img_correlacao = np.zeros(np.array(img.shape) + np.array(mascara.shape) - 1)

  for x in range(mascara.shape[0]):
    for y in range(mascara.shape[1]):

      img_correlacao[x:x+img.shape[0], y:y+img.shape[1]] += img * mascara[x,y]

      return img_correlacao.astype('uint8')

img = cv.imread('/content/img14.jpg', 0)
mascara = np.array([ [1,1,1], [1,1,1], [1,1,1] ])

imagemCorrelacao = correlacao(img, img_corr, mascara)
cv2_imshow(imagemCorrelacao)

print('')
print(imagemCorrelacao)

    # Convolução
def rotate_180(image):

    size_image = image.shape
    M = size_image[0]
    N = size_image[1]

    new_image = np.zeros((M, N))

    for i in range (0, M):
        for j in range(0, N):
            new_image[(M-1)-i, (N-1)-j] = image[i,j]

    return new_image

def get_slice(point, size, image):

    delta = int(size/2)

    a = (point[0] - delta)
    b = (point[1] - delta)

    return image[a:(a+size),b:(b+size)]

def convolucao(imagem, mascara):
    size_imagem = imagem.shape

    M = size_imagem[0]
    N = size_imagem[1]

    size_mascara = mascara.shape

    m = size_mascara[0]
    n = size_mascara[1]

    nova_imagem = np.zeros(imagem.shape)

    delta = int(m/2)

    for i in range (0+delta, M-delta):
        for j in range(0+delta, N-delta):
            slice_imagem = get_slice((i,j),m,imagem)
            mascara_180 = rotate_180(mascara)
            new_i_j = np.sum(slice_imagem * mascara_180)
            nova_imagem[i,j] = new_i_j

    return nova_imagem

mascara = np.array([[-1, 0, 1], [-2, 1, 2], [-1, 0, 1]])
imagemConvolucao = convolucao(imgOriginal,mascara)
cv2_imshow(imagemConvolucao)

print('')
print(imagemConvolucao)

  # Filtragem no Domínio da Frequência

    # Transformada de Fourier
img = cv.imread('/content/img15.jpg',0)
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Imagem original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Imagem do Espectro de Magnitude')
plt.xticks([])
plt.yticks([])
plt.show()

rows, cols = img.shape
crow,ccol = rows//2 , cols//2

mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.figure(figsize=(11,6))
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Imagem original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(img_back, cmap = 'gray')
plt.title('Filtro passa-baixa')
plt.xticks([])
plt.yticks([])
plt.show()

    # Transformada Inversa de Fourier

MN = 300
imgT = np.zeros((MN, MN, 1), np.uint8)

x = y = int(MN/2)

raio = 30

I,J = np.meshgrid(np.arange(imgT.shape[0]),np.arange(imgT.shape[1]))

distancia = np.sqrt((I-x)**2+(J-y)**2)

imgT[np.where(distancia<raio)] = 255
img_fourier = np.fft.fftshift(imgT)

print("Imagem no domínio da frequência")
cv2_imshow(imgT)

print("Transformada Inversa de Fourier para voltar ao domínio espacial")
cv2_imshow(img_fourier)
