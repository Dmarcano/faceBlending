# from .face_coordinates import detec_face_coordinates
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
# from .face_coordinates import detect_face_coordinates
from face_coordinates import detect_face_coordinates
import unittest
import numpy as np 
import cv2 



class Test_FaceDetectionAPI(unittest.TestCase):


    def test_true_negative(self):
        img = cv2.imread('../../images/obama.jpg')

        if not isinstance(img, np.ndarray):
            raise Exception("Failed to load image!")

        zero_arr = np.zeros_like(img)
        result = detect_face_coordinates(zero_arr)

        self.assertEqual((), result)

    def test_true_positive(self):
        img = cv2.imread('../../images/obama.jpg')
        if not isinstance(img, np.ndarray):
            raise Exception("Failed to load image!")
        result = detect_face_coordinates(img)

        self.assertEqual(len(result), 1) # obama picture has one result
        return

    def multiple_faces(self):
        

        return


if __name__ == "__main__":
    import os 
    # Change working directory to file directory for debbugging
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()