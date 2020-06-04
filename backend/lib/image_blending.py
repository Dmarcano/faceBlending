"""
This module pertains to the generic blending of two images using Laplacian pyramids. 
"""

import cv2 
import numpy as np 

class DimensionMisMatchException(Exception):
    """
    Custom exception when a face cannot be found
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def pyr_build(img, num_levels):
    """
    given an image makes an num_levels + 1 laplacian pyramid of it
    """

    # * variables to store laplacian pyramid and G used to build it 
    lap_pyr = []
    curr_g = img.copy()
    h, w = curr_g.shape[:2]
    
    for i in range(num_levels):
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

def pyr_build_max(img):
    """
    given an image makes a laplacian pyramid of the utmost fidelity. 

    This is ussually overkill and can result in some errors due to continually reducing an image by a scaling factor. 


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

def alpha_blend(A, B, alpha):
    # 
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad# out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = np.expand_dims(alpha, 2)
        
        return A + alpha*(B-A)



def combine_images_pyramid(image1, image2, mask, num_levels, copy = False):
    """
    given two images and an alpha mask,
    it blends the two images along an arbitrary mask given

    Requires that the dimensions of the 
    """

    if image1.shape != image2.shape:
        raise DimensionMisMatchException(f"Picture Dimensions of image1 : {image1.shape} and  image2: {image2.shape} do not match")

    img1,img2 = image1.copy(), image2.copy() if copy else image1, image2

    # get the laplacian pyramids of each image
    img1_lap_pyr = pyr_build(img1, num_levels)
    img2_lap_pyr = pyr_build(img2, num_levels)

    if len(img1_lap_pyr)!= len(img2_lap_pyr):
        print(f"Laplacian pyramids are mismtached! lengths: L1:{len(img1_lap_pyr)}  vs L2:{len(img2_lap_pyr)} Results may vary!")
    
    mixed_pyramid = []
    for i in range(len(img1_lap_pyr)):
        # resize alpha mask
        h,w = img1_lap_pyr[i].shape[:2]
        resized_alpha = cv2.resize(mask, (w,h), interpolation=cv2.INTER_AREA)
        # blend and append to new pyramid
        mixed_pyr_level = alpha_blend(img1_lap_pyr[i], img2_lap_pyr[i], resized_alpha)

        mixed_pyramid.append(mixed_pyr_level)

    combined_image = pyr_reconstruct(mixed_pyramid)
    return combined_image