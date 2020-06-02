import cv2 
import numpy as np 

def add_edges():
    img = cv2.imread('../../images/monster.png')
    # img = cv2.imread('../../images/monster.png', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img,50,70)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # edges *= 
    add = cv2.addWeighted(img, 1, edges, 0.5, 0.0)
    cv2.imshow('shit', img)
    cv2.imshow('two', edges)
    cv2.imshow('added', add)
    cv2.waitKey()

