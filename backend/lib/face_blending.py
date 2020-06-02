"""
This module implements a model and mechanism for easily blending parts of different faces

"""
from face_coordinates import detect_features, draw_bounding_boxes, draw_facial_features, FaceNotFoundError
import numpy as np 
import cv2 


class FaceBlendingModel():
    """
    A container for an image, and all of its facial features. 
    """

    def __init__(self, img):
        self.face_found = False 
        self.img = self.resize(img,720) 
        try:
            self.bounding_rect, landmark_points = detect_features(self.img)
            self.jaw = landmark_points[0:16]
            self.left_eyebrow = landmark_points[17:21]
            self.right_eyebrow = landmark_points[22:26]
            self.nose = landmark_points[27:35]
            self.left_eye = landmark_points[36:41]
            self.right_eye = landmark_points[42:47]
            self.mouth = landmark_points[48:67]

            self._feature_dict = {
                'jaw' : self.jaw,
                'left_eyebrow' : self.left_eyebrow,
                'right_eyebrow' : self.right_eyebrow, 
                'nose' : self.nose,
                'left_eye' : self.left_eye,
                'right_eye' : self.right_eye,
                'mouth' : self.mouth
            }

            #  now with the landmark points we make each face feature the following

        except FaceNotFoundError:
            print('There was a problem finding a face! try another photo')

    def resize(self, img, width):
        img_h, img_w, _ = img.shape
        ratio = width/img_w
        dim = (width, int(img_h * ratio))
        return cv2.resize(img, dim)

    def draw_feature(self, feature):
        if feature not in self._feature_dict:
            raise Exception('must use the following features: ')
        
        copy = self.img.copy()
        facial_coordinates = self._feature_dict[feature]
        return draw_facial_features(copy,facial_coordinates)
             
    
    def test(self, img):
        
        print(f"this is a test! face is {self.face_found}")


def alpha_blend(A, B, alpha):
    # 
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad# out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = np.expand_dims(alpha, 2)
        
        return A + alpha*(B-A)

def pyr_build(img):
    """
    given an image makes an num_levels + 1 laplacian pyramid of it
    """

    # * variables to store laplacian pyramid and G used to build it 
    lap_pyr = []
    curr_g = img.copy()
    h, w = curr_g.shape[:2]
    
    while h > 16 and w > 16:
        # * 1) G(i+1) = G(i) convolved and blurred
        next_g = cv2.pyrDown(curr_g)
        h, w = curr_g.shape[:2]
        # * 2) Upscale G(i+1)
        next_g_up = cv2.pyrUp(next_g, dstsize=(w,h))
        # * L(i+1) is equal to G(i) - upscale G(i+1)
        laplacian =  curr_g.astype(np.float32) - next_g_up.astype(np.float32)
        lap_pyr.append(laplacian)

        curr_g  = next_g
        
        laplacian_to_show = laplacian
        # cv2.imshow('pyramid', 0.5 + 0.5*(laplacian_to_show / np.abs(laplacian_to_show).max()))
        # cv2.waitKey()
    # set final laplacian equal to final g
    lap_pyr.append(next_g)

    return lap_pyr 

def test():
    path = "../../images/obama.jpg"
    img = cv2.imread(path)
    zero = np.zeros_like(img)

    face_model = FaceBlendingModel(img)
    
    for feature in face_model._feature_dict.keys():
        img = face_model.draw_feature(feature)
        cv2.imshow('test', img)
        cv2.waitKey()

    return 

if __name__ == "__main__":
    import os 
    # Change working directory to file directory for debbugging
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    test()
    