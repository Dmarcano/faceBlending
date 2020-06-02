"""
This module allows you to get the coordinates of a face and its features
in an image using Open CV and Dlib 

credits for the helper function from img utils

"""
import cv2
import numpy as np
from os import chdir, listdir
from os.path import abspath, isfile, join, dirname
from glob import glob 
import dlib



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

def __get_facial_feature_detector():
    dname = dirname(__file__)
    xml_files = glob(f'{dname}/xmlWeights/*.dat')
    filename = 'shape_predictor_68_face_landmarks'

    if len(xml_files) == 0:
        raise FileNotFoundError("Failed to find XML weights for face detection inside xmlWeights folder")

    hits = list(filter(lambda path :filename  in path, xml_files))
    
    
    detector = dlib.shape_predictor(hits[0])

    return detector

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords



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

def draw_facial_features(img, feature_coords):
    """
    Draws the 68 feature points 
    """

    for (x,y) in feature_coords:
       cv2.circle(img, (x,y), 1, (0,0,255), -1)


    return img 

def detect_features(img : np.ndarray):
    """
    Function which detects all facial pictures of a single person in the picture.
    
    For more than one person use detect_features_many()

    img : numpy.ndarray -> image to detect the features in 
    """
    # get the detector 
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    predictor = __get_facial_feature_detector()


    people = []
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        bounding_rect = rect_to_bb(rect)
        person = (bounding_rect, shape)
        
        return person 
        



def _test():
    img = cv2.imread('../../images/monster.png')
    h,w, _ = img.shape
    # img = cv2.resize(img, (w//4, h//4))
    if not isinstance(img, np.ndarray):
        raise Exception("Failed to load image!")
    # give us a person with a bounding rectangle for 0 idx and 68 face features for idx 1
    person = detect_features(img)
    
    img = draw_bounding_boxes(img, [person[0]])
    img = draw_facial_features(img, person[1])
    
    cv2.imshow("test", img)
    cv2.waitKey()

    return 





if __name__ == "__main__":
    # Change working directory to file directory for debbugging
    abspath = abspath(__file__)
    dname = dirname(abspath)
    chdir(dname)
    _test()