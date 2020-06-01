"""
This module allows you to get the coordinates of a face and its features
in an image using simple face detection found in open CV

"""
import cv2
import numpy as np
from os import chdir, listdir
from os.path import abspath, isfile, join, dirname
from glob import glob 

SUPPORTED_FEATURES = {  "eyes" : "haarcascade_eye_tree_eyeglasses.xml", 
                        'left_eye' : 'haarcascade_lefteye_2splits.xml', 
                        'right_eye': "", 
                        'smile' : "", 
                        "face" : "haarcascade_frontalface_alt.xml",
                        "face_profile": "haarcascade_profileface.xml"}

def __get_haar_cascade_classifier(xml_weights_name :str ="haarcascade_frontalface_alt.xml" ):
    """
    helper function that searches directories for a haar_cascade xml weights and instantiates 

    xml_weights_path : name of the xml weights found in the xmlWeights folder
    """
    # get current direcotry and ask glob to find all xml files inside of it
    dname = dirname(__file__)
    xml_files = glob(f'{dname}/xmlWeights/*.xml')

    if len(xml_files) == 0:
        raise FileNotFoundError("Failed to find XML weights for face detection inside xmlWeights folder")
    
    hits = list(filter(lambda path :xml_weights_name  in path, xml_files))
    # grab the first xml cascade classifier weights
    classifier = cv2.CascadeClassifier(hits[0])

    return classifier

def detect_face_coordinates(img : np.ndarray, face_detector = None):
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

def draw_bounding_boxes(img, coordinates_list):

    for coords in coordinates_list:
        x,y,w,h = coords
        cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)

    return img

def detect_features(img : np.ndarray, feature : str):
    """
    Function which detects specific facial features

    img : numpy.ndarray -> image to detect the features in 

    feature : string -> a feature for which to look for. must be either ('eyes', 'face')
    
    """

    if feature not in SUPPORTED_FEATURES:
        raise NotImplementedError(f'specified feature is not implemented!\nthe only implemented keywords are {SUPPORTED_FEATURES}')
    classifier = __get_haar_cascade_classifier(xml_weights_name= SUPPORTED_FEATURES[feature] )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, 1.1, 4)

    return features

def _test():
    img = cv2.imread('../../images/obama.jpg')
    h,w, _ = img.shape
    img = cv2.resize(img, (w//4, h//4))
    if not isinstance(img, np.ndarray):
        raise Exception("Failed to load image!")
    features = ['eyes']

    for feature in features:
        result = detect_features(img, feature)
        draw_bounding_boxes(img, result)
    
    cv2.imshow("test", img)
    cv2.waitKey()

    return 

if __name__ == "__main__":
    # Change working directory to file directory for debbugging
    abspath = abspath(__file__)
    dname = dirname(abspath)
    chdir(dname)

    _test()