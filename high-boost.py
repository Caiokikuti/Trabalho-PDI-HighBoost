import cv2
import sys
import numpy as np


def subtracao(img1, blur):
    height = img1.shape[0]-1
    width = img1.shape[1] -1
    if img1.shape == blur.shape:
        image = img1.copy()
        image[:img1.shape[0]-1][:img1.shape[1] -1] = abs(img1[:img1.shape[0]-1][:img1.shape[1] -1] - blur[:img1.shape[0]-1][:img1.shape[1] -1])
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


def main():
    #Abrir imagem 
    if (len(sys.argv)!=2): 
        print("Error, Try like this: python boxfilter.py <nome_do_arquivo_de_imagem.png> <TAXA DE REDUÇÃO>")
        print("OBS: As imagens da paisagem e do gato sao .jpg, o nome da imagem deve sempre conter o formato da extensao junto!!")
        sys.exit()

    nomeImagem = str(sys.argv[1])
    img1 = cv2.imread(nomeImagem, 0)
    cv2.imshow('image', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    # realizadno o processo de blurr

    blurr = cv2.blur(img1, (5,5))
    # mask = subtracao(img1, blurr)
    mask = cv2.subtract(img1, blurr)
    add = cv2.add(img1, mask)

    cv2.imshow('image', add)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    


if __name__ == "__main__":
    main()