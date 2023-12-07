import os
import cv2
import numpy as np
import subprocess
import torch
from torchvision.utils import flow_to_image

# internal imports
from alignImages import alignImages
from composite import calc_Mflow, alpha_blending
from raft import calculateRaftOpticalFlow

"""
Long Exposure Pipeline

1. read all images

2. align images using the first frame as the reference

3. subject detection (creating the face mask) -> one face mask

4. interpolate between frames -> one blurred image

5. composite
"""


def readImages(directory):
    """
    Read and return a list of images from directory

    Returns:
        List of images
    """
    images = []
    for filename in os.listdir(directory):
        if filename == "aligned_images":
            continue
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)
        images.append(img)
    return images


def writeImages(directory, images):
    """
    Write images to directory
    """
    num_digits = 3
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(directory, f"img_{i:0>{num_digits}}.png"), img)


def getAlignedImaged(directory, images):
    # Read in cached aligned iimages if exists
    if len(os.listdir(directory)) != 0:
        alignedImages = []
        for filename in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, filename))
            alignedImages.append(img)

    # Else generate aligned images and save to directory
    else:
        alignedImages = alignImages(images)
        writeImages(directory, alignedImages)
        
    return alignedImages


def calculateOpticalFlow(images, method, from_cache=False, flowmap_dir=None):
    if flowmap_dir:
        flowmap_dir = os.path.join(flowmap_dir, method)
    
    # load from cache if possible
    if from_cache and flowmap_dir and len(os.listdir(flowmap_dir)) == len(images)-1:
        flowmaps = []
        for filename in os.listdir(flowmap_dir):
            flowmap = np.load(os.path.join(flowmap_dir, filename))
            flowmaps.append(flowmap)
        return flowmaps
    
    if method == 'cv2':
        # convert images to gray scale
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
        
        # calculate pair-wise optical flow maps
        cv2flow = lambda img1, img2: cv2.calcOpticalFlowFarneback(img1, img2, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
        flowmaps = [cv2flow(gray_images[i-1], gray_images[i]) for i in range(1, len(gray_images))]
    
    elif method == 'raft':
        # calculate pair-wise optical flow maps
        flowmaps = calculateRaftOpticalFlow(images)
        
    else:
        flowmaps = []
    
    # cache if possible
    if flowmap_dir:
        for i, flowmap in enumerate(flowmaps):
            np.save(os.path.join(flowmap_dir, f"{method}_flowmap_{i:0>{3}}"), flowmap)

    return flowmaps


def subjectDetection(image):
    """ 
    From an image, produces and returns a face mask

    Returns:
        Face mask
    """
    # TODO: Maybe also allow user to use their own custom mask

    # TODO: Find gaze attention mask
    # TODO: Find subject mask
    # TODO: Combine both of them according to equation in paper
    face_mask = None
    return face_mask


def interpolateFrames(frame1, frame2, num_frames):
    """
    Takes in 2 sequential frames, produces a list of frames in between
    
    Returns:
        List of inbetween frames
    """
    inbetween_frames = None
    return inbetween_frames


def blurImages(images):
    """
    Takes a series of images and blurs them together via linear interpolation.
    
    Returns:
        One blurred image
    """
    blurred_image = images[0]
    weight = 1
    for i in range(1, len(images)):
        # generate inbetween
        NUM_FRAMES = 10
        inbetween_frames = interpolateFrames(images[i-1], images[i], NUM_FRAMES)
        inbetween_frames.append(images[1])
        new_weight = weight + len(inbetween_frames)
        
        # blur images
        blurred_image = (weight * blurred_image + np.sum(np.stack(inbetween_frames, -1), axis=-1)) / new_weight
        weight = new_weight
    return blurred_image


def composite(sharp_image, blurred_image, flow_maps, subject_mask):
    """
    Composite between MFlow, subject mask, sharp, and blurred to return final output image 
    
    Returns:
        Final output image
    """
    MFlow = calc_Mflow(flow_maps, sharp_image)

    # combine the flow and the clipped face masks with a simple max operator
    flow_face_mask = np.where(MFlow > subject_mask, MFlow, subject_mask)
    
    # use composite function(s) from previous previous project code
    composite = alpha_blending(sharp_image, flow_face_mask, blurred_image)

    return composite
     

def pipeline():
    image_directory = "examples/helen"
    flowmap_directory = "flowmap"

    # 1. read all images
    images = readImages(image_directory)
    print(f"number of images: {len(images)} = N")

    # # 2. align images using the first frame as the reference
    # aligned_images_directory = "examples/helen/aligned_images"
    # images = getAlignedImaged(aligned_images_directory, images)

    # 2. read/calculate optical flow maps
    method = "raft"
    flow_maps = calculateOpticalFlow(images, method=method, from_cache=False, flowmap_dir=flowmap_directory)
    print(f"number of flow maps: {len(flow_maps)} = N-1")
    print(f"flow shape: {flow_maps[0].shape} = (H, W, 2)")
    print()

    flow_img = flow_to_image(torch.tensor(flow_maps[0].transpose(2, 0, 1), dtype=torch.float32)).numpy().transpose(1, 2, 0)
    print(flow_img.shape)
    cv2.imshow("example flow map", flow_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 3. subject detection (creating the face mask) -> one face mask
    sharp_image = images[0]
    face_mask = subjectDetection(sharp_image)
    cv2.imwrite("output/face_mask.png", face_mask)

    # 4. interpolate between frames -> one blurred image
    blurred_image = blurImages(images)
    cv2.imwrite("output/blurred_image.png", blurred_image)

    # 5. composite
    result = composite(sharp_image, blurred_image, flow_maps, face_mask)
    cv2.imwrite("output/result.png", result)


if __name__ == "__main__":
    pipeline()