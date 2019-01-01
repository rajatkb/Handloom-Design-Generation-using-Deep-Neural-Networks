import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


def create_mask(image , channel = 1):
    '''
    image: numpy format image of hxwx1 size 
    '''
    (thresh, img) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ind_w = img > 128
    ind_b = img < 128
    img[ind_w] = 0
    img[ind_b] = 255
    if channel == 3:
        img = np.stack((img,)*3,-1)
    return img

def get_masks(image):
    mask = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
    mask = create_mask(mask)
    inverted_mask = cv2.bitwise_not(mask)
    return mask , inverted_mask

def merge(image , net , dir , dim = 512 ,  weights = []):
    mask , inv_mask = get_masks(image)
#     print(image.shape , mask.shape)
    assert(weights != [])
    net.load_weights(dir+weights[0]+".h5")
    res1 = np.array(net.predict(np.array([image])) , dtype = np.uint8)[0]
    net.load_weights(dir+weights[1]+".h5")
    res2 = np.array(net.predict(np.array([image])) , dtype = np.uint8)[0]
#     print(res1.shape , mask.shape)
    masked1 = cv2.bitwise_and(res1 , res1 , mask = inv_mask)
    masked2 = cv2.bitwise_and(res2 , res2 , mask = mask)
    return np.array((masked1+masked2), dtype = np.uint8)