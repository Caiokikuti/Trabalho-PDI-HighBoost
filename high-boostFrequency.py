import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from PIL import Image
import scipy.fftpack as fp
import scipy.signal as signal


def subtracao(img1, blur):
    height = img1.shape[0]-1
    width = img1.shape[1] -1
    if img1.shape == blur.shape:
        image = img1.copy()
        image[:img1.shape[0]-1][:img1.shape[1] -1] = img1[:img1.shape[0]-1][:img1.shape[1] -1] - blur[:img1.shape[0]-1][:img1.shape[1] -1]
        return image
    else:
        print('error - imagens nao sao do mesmo tamanho')

def adicao(img1, sub):
    height = img1.shape[0]-1
    width = img1.shape[1] -1
    if img1.shape == sub.shape:
        image = img1.copy()
        image[:img1.shape[0]-1][:img1.shape[1] -1] = img1[:img1.shape[0]-1][:img1.shape[1] -1] + sub[:img1.shape[0]-1][:img1.shape[1] -1]
        return image
    else:
        print('error - imagens nao sao do mesmo tamanho')

def highBoostFilter(img, order, corte, a):
    dft = np.fft.fft(img)
    sdft = np.fft.fftshift(dft)

    magnitude = sdft

    row, col = img.shape
    HP = np.zeros(img.shape)
    multiply = np.zeros(img.shape)

    crows, ccols = int(row/2), int(col/2)

    for i in range(-crows ,crows):
        for j in range(-ccols, ccols):
            distancia = math.sqrt(pow(i,2)+pow(j,2))
            one = corte if distancia == 0  else (corte/distancia)
            two = 2*order
            demo = 1+pow(one, two)
            HP[i,j] = (a-1)+(1/demo)
    
    mask = np.ones((row, col), np.uint8)
    r = 80
    center = [crows, ccols]
    x, y = np.ogrid[:row, :col]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0 
    # mask = cv2.GaussianBlur(img, (11, 11), cv2.BORDER_DEFAULT)
    # gauss_kernel = np.outer(3, signal.butter(img.shape[0], 'low'), signal.butter( 3, img.shape[1], 'low'))

    HP = fp.fft2(fp.ifftshift(HP))

    magnitude = np.multiply(magnitude, HP)
   
    shiftBack = np.fft.ifftshift(multiply)
    back = np.fft.ifft(shiftBack)
    final_image = np.abs(back)
    final_image = final_image * 255 / final_image.max()
    # final_image = final_image.astype(np.uint8)
    x=np.array(final_image, dtype=np.uint8)

    return x
    


def main():
    #Abrir imagem 
    if (len(sys.argv)!=2): 
        print("Error, Try like this: python boxfilter.py <nome_do_arquivo_de_imagem.png> <TAXA DE REDUÇÃO>")
        print("OBS: As imagens da paisagem e do gato sao .jpg, o nome da imagem deve sempre conter o formato da extensao junto!!")
        sys.exit()

    nomeImagem = str(sys.argv[1])
    img1 = cv2.imread(nomeImagem, 0)
    add = highBoostFilter(img1,1,50,1.5)
    # add = highBoost1(img1, 80)
    cv2.imshow('sem filtro',add)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = cv2.subtract(img1, add)
    appMask = cv2.add(img1, mask)


    cv2.imshow('image', appMask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    


if __name__ == "__main__":
    main()