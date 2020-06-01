"""
This module allows you to get the coordinates of a face in an image using simple face detection found in open CV

"""
import cv2
import numpy as np
from os import chdir, listdir
from os.path import abspath, isfile, join, dirname
from glob import glob 

def __get_haar_cascade_classifier():
    """
    helper function that searches directories for a haar_cascade xml weights and instantiates 
    """
    # get current direcotry and ask glob to find all xml files inside of it
    dname = dirname(__file__)
    xml_files = glob(f'{dname}/*.xml')

    if len(xml_files) == 0:
        raise FileNotFoundError("Failed to find XML weights for face detection inside lib folder")
    
    hits = list(filter(lambda path : "haarcascade_frontalface_alt.xml" in path, xml_files))
    # grab the first xml cascade classifier weights
    classifier = cv2.CascadeClassifier(hits[0])

    return classifier

def detect_face_coordinates(img, face_detector = None):
    """
    Function which given an image, detects if there are any faces in the image, 
    if there are any faces, all their coordinates are returned in a list of tuples

    each tuple contains (x,y,w,h)

    @param img -> an image as a numpy array

    @param face_detector (optional) -> reference to an open cv haarcascade face detector. If none is provided then a new one is instatiated

    """

    # if there is no face detector then instantiate one from open CV
    classifier = __get_haar_cascade_classifier() if face_detector == None else face_detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.1, 4)

    return faces

def _test():
    img = cv2.imread('../images/obama.jpg')
    if not isinstance(img, np.ndarray):
        raise Exception("Failed to load image!")
    result = detect_face_coordinates(img)

    return 

if __name__ == "__main__":
    # Change working directory to file directory for debbugging
    abspath = abspath(__file__)
    dname = dirname(abspath)
    chdir(dname)

    _test()