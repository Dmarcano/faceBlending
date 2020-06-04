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

    def __init__(self, img, width = 720):
        self.face_found = False 
        self.img = self.resize(img,width) 
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

    def draw_feature(self, *features, img =None):
        """
        function which draws a facial feature overtop an image. 

        The supported features are those features for facial landmark detection and full_face for all features at once

        """

        if isinstance(img, type(None)):
            raise Exception("please provide an image")

        feature_list = self._feature_dict.keys() if "full_face" in features else [feature for feature in features]

        for face_feature in feature_list:
            # checks if the passed feature is part of the supported feature
            if face_feature not in self._feature_dict:
                raise Exception(f"Please provide one of the following features:\n {self._feature_dict.keys()} or provide 'full_face'")

            if face_feature == 'jaw':
                pass 

            else:
                coords = self._feature_dict[face_feature]
                hull = cv2.convexHull(coords)
                cv2.drawContours(img, [hull], -1, (255,255,255), -1)  
                      

        return img

    def resize(self, img, width):
        """
        resises the image to a specific width while mainting the aspect ratio of the picture
        """
        img_h, img_w, _ = img.shape
        ratio = width/img_w
        dim = (width, int(img_h * ratio))
        return cv2.resize(img, dim)

    def extend(self, height):
        """
        returns an extended version of the base image.
        Adds more blank rows on the bottom of the image to reach a specific height
        """
        h, w, _ = self.img.shape 
        rows = np.array([np.zeros((500,3)) for row in range(height) ])
        extended_copy = np.vstack((self.img.copy(), rows))
        print(extended_copy.shape)
        return

    def crop(self, height):
        """
        returns a cropped version of the base image such that 
        """
        return 

    def draw_feature_points(self, feature):
        """
        draws points for facial features
        """
        if feature not in self._feature_dict:
            raise Exception('must use the following features: ')
        
        copy = self.img.copy()
        facial_coordinates = self._feature_dict[feature]
        return draw_facial_features(copy,facial_coordinates)
             

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

def pyr_reconstruct(lap_pyramid):
    """
    Given a laplacian pyramid. Rebuilds the base image from it
    """

    laplacians = lap_pyramid.copy()
    reconstruct_list = [laplacians.pop()]
    # reverse the list so everything goes from 0 to size
    laplacians.reverse()
    num_levels = len(laplacians)

    for i in range(num_levels):
        current_r = reconstruct_list[i]
        # next Ri = curr Ri upscaled + next Laplacian  
        next_laplacian = laplacians[i]
        h,w = next_laplacian.shape[:2]
        # upscale current r and add the next laplacian to it
        pyrup = cv2.pyrUp(current_r, dstsize=(w,h))
        next_r =  pyrup + next_laplacian
        reconstruct_list.append(next_r)

    to_return = reconstruct_list.pop()
    # to_return = to_return - np.amin(to_return)
    to_return = (to_return / np.abs(to_return).max())
    
    #cv2.imshow('rebuild as float', (to_return / np.abs(to_return).max()))
    to_return = np.clip(to_return, 0, 1)
    to_return = to_return*255
    to_return = to_return.astype(np.uint8)

    return to_return


def extend_test():
    path = "../../images/monster.png"
    img = cv2.imread(path)
    zero = np.zeros_like(img)

    face_model = FaceBlendingModel(img, width= 500)
    face_model.extend(10)

    return

def facial_points_test():
    path = "../../images/monster.png"
    img = cv2.imread(path)
    zero = np.zeros_like(img)

    face_model = FaceBlendingModel(img, width= 500)

    for feature in face_model._feature_dict.keys():
        img = face_model.draw_feature_points(feature)
        cv2.imshow('test', img)
        cv2.waitKey()

    return

def test():
    path = "../../images/monster.png"
    img = cv2.imread(path)
    zero = np.zeros_like(img)

    face_model = FaceBlendingModel(img, width= 500)
    
    to_show = face_model.draw_feature("full_face" , img = np.zeros_like(face_model.img))
    second = face_model.draw_feature("left_eye", 'mouth' , img = face_model.img.copy())
    img = face_model.draw_feature('full_face', img= face_model.img.copy())

    cv2.imshow("shit" , to_show)
    cv2.imshow("second" , second)

    cv2.imshow('sorry', img)
    cv2.waitKey()

    return 

if __name__ == "__main__":
    import os 
    # Change working directory to file directory for debbugging
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    # facial_points_test()
    extend_test()
    