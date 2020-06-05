"""
This module implements a model and mechanism for easily blending parts of different faces

"""
from face_coordinates import detect_features, draw_bounding_boxes, draw_facial_features, FaceNotFoundError
from image_blending import combine_images_pyramid, DimensionMisMatchException
import numpy as np 
import cv2 


class FaceBlendingModel():
    """
    A container for an image, and all of its facial features. 
    """

    def __init__(self, img, width = 720):
        self.face_found = False 
        self.img : np.ndarray = self.resize(img,width) 
        try:
            bounding_rect, landmark_points = detect_features(self.img)
            self.bounding_rect = bounding_rect
            self.jaw = landmark_points[0:17]
            self.left_eyebrow = landmark_points[17:22]
            self.right_eyebrow = landmark_points[22:27]
            self.nose = landmark_points[27:36]
            self.left_eye = landmark_points[36:42]
            self.right_eye = landmark_points[42:48]
            self.mouth = landmark_points[48:68]

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

    def draw_feature(self, *features, img =None, dtype = np.uint8):
        """
        function which draws a facial feature overtop an image. 

        The supported features are those features for facial landmark detection and full_face for all features at once

        """

        if isinstance(img, type(None)):
            raise Exception("please provide an image")

        feature_list = self._feature_dict.keys() if "full_face" in features else [feature for feature in features]

        pixel_intensity = {
            np.uint8 :  (255,255,255),
            np.float32 : (1,1,1)
        }

        for face_feature in feature_list:
            # checks if the passed feature is part of the supported feature
            if face_feature not in self._feature_dict:
                raise Exception(f"Please provide one of the following features:\n {self._feature_dict.keys()} or provide 'full_face'")

            if face_feature == 'jaw':
                pass 

            else:
                coords = self._feature_dict[face_feature]
                hull = cv2.convexHull(coords)
                cv2.drawContours(img, [hull], -1, pixel_intensity[dtype], -1)  
                      

        return img

    def resize(self, img, width):
        """
        resises the image to a specific width while mainting the aspect ratio of the picture
        """
        img_h, img_w, _ = img.shape
        ratio = width/img_w
        dim = (width, int(img_h * ratio))
        return cv2.resize(img, dim)

    def extend(self, num_rows, dtype = np.uint8):
        """
        returns an extended version of the base image.
        Adds more blank rows on the bottom of the image to reach a specific height

        @param num_rows: integer -> number of rows to add to the image
        
        @param dtype : -> dtype. a datatype that is supported by numpy eg (np.float64 etc.)
        """
        h, w, _ = self.img.shape 
        rows = np.array([np.zeros((500,3), dtype=dtype) for row in range(num_rows) ])
        extended_copy = np.vstack((self.img.copy(), rows))
        return extended_copy

    def crop(self, height):
        """
        returns a cropped version of the base image such that 
        """
        return 

    def draw_feature_points(self, feature, dtype = np.uint8):
        """
        draws points for facial features
        """
        if feature not in self._feature_dict:
            raise Exception('must use the following features: ')
        
        copy = self.img.astype(dtype = dtype)
        facial_coordinates = self._feature_dict[feature]
        return draw_facial_features(copy,facial_coordinates)
    
    def get_facial_feature_array(self):
        """
        returns an ndarray with all 68 points from the dlib facial landmark detection algorithm
        """
        to_return = np.concatenate( [arr for arr in self._feature_dict.values()] )

        return to_return

    def get_face_bounding_box(self, dtype = np.int32):
        """
        Gets an np array with the face bounding box that was found by the dlib face detection module

        it has three points that correspond to the top-left, top-right, bottom-right points of the bounding box
        """
        x,y,w,h = self.bounding_rect
        return np.array([ [x,y], [x + w, y], [x + w, y + h] ] , dtype=dtype  )

    def get_all_face_points(self):
        """
        Gets one array with both the bounding box points and facial landmark points
        """
        return np.concatenate((self.get_face_bounding_box(), self.get_facial_feature_array()))

    def get_all_facial_feature_labels(self):
        """
        returns a list of all the feature labels that are used for face detection
        """
        return list(self._feature_dict.keys())



def blend_face_models(src_face_model : FaceBlendingModel, dst_face_model : FaceBlendingModel, *features):
    """
    given two face models and a set of features to blend, then creates an output image which is the blended 
    """
    assert isinstance(src_face_model, FaceBlendingModel)
    assert isinstance(dst_face_model, FaceBlendingModel)

    src_h, src_w, _ = src_face_model.img.shape
    dst_h, dst_w, _ = dst_face_model.img.shape
    # make sure the widths of the models match 
    if src_w != dst_w:
        raise DimensionMisMatchException(f"Source model image width of {src_w} does not match destination model image width of {dst_w}")
    if len(features) == 0:
        raise Exception("No features given to function")

    __align_face_models(src_face_model, dst_face_model)
    
    destination_extended = False 
    # make sure that the heights of the models match 
    if src_h > dst_h:
        difference = src_h - dst_h
        src_img = src_face_model.img.copy()
        dst_img = dst_face_model.extend(difference)
        destination_extended = True 

    elif src_h <dst_h:
        difference =  dst_h - src_h
        src_img = src_face_model.img.extend(difference)
        dst_img = dst_face_model.copy()
        
    else:
        src_img = src_face_model.img.copy()
        dst_img = dst_face_model.copy()

    mask = src_face_model.draw_feature(*features, img =np.zeros((src_h, src_w, _),dtype=np.float32), dtype=np.float32)
    
    blended_img = combine_images_pyramid(src_img = src_img, dst_img= dst_img, mask = mask, num_levels=8)

    if destination_extended:
        blended_img = blended_img[0 : dst_h, : ]

    return blended_img


def __align_face_models(src_face_model : FaceBlendingModel, dst_face_model : FaceBlendingModel):
    """
    returns an aligned version of the 
    """
    assert isinstance(src_face_model, FaceBlendingModel)
    assert isinstance(dst_face_model,FaceBlendingModel)
    # need to get the affine transform from the source image to the destination image, this can be done by using the bounding box points and 
    # face feature points 

    src_points = src_face_model.get_all_face_points()
    dst_points = dst_face_model.get_all_face_points()
    
    return 


def blend_faces(src_face, dst_face, width, *features, **meta_data):

    if len(features) == 0:
        raise Exception("No features given to function")

    src_model = FaceBlendingModel(src_face, width = width)
    dst_model = FaceBlendingModel(dst_face, width = width)

    combined_img = blend_face_models(src_model, dst_model, *features)
    
    

    return combined_img


def extend_test():
    path = "../../images/obama.jpg"
    img = cv2.imread(path)
    zero = np.zeros_like(img)

    face_model = FaceBlendingModel(img, width= 500)
    img = face_model.img.copy()
    img = face_model.extend(100)
    face_model.draw_feature("full_face", img = img)

    cv2.imshow('shh', img )
    cv2.waitKey()


    return

def face_blend_test_full_features():
    path1 = "../../images/obama.jpg"
    path2 = "../../images/tyson.jpg"

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    tyson_model = FaceBlendingModel(img1, width=500)
    img_cutout = tyson_model.draw_feature('full_face', img= tyson_model.img.copy())
    combined_img = blend_faces(img1, img2, 500, "full_face")

    cv2.imshow("result", combined_img)
    cv2.imshow("sorry mom", img_cutout)
    cv2.waitKey()


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
    path = "../../images/tyson.jpg"
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
    # extend_test()
    # test()
    face_blend_test_full_features()
    