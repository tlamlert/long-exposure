import os
import cv2
import numpy as np
import subprocess

from alignImages import alignImages
from composite import calc_Mflow, alpha_blending

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
        img = cv2.imread(os.path.join(directory, filename))
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


def calculateOpticalFlow(images, method, image_directory=None, flowmap_directory=None):
    if method == 'cv2':
        # convert images to gray scale
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
        
        # calculate pair-wise optical flow maps
        flow_maps = []
        for i in range(1, len(gray_images)):
            flow_cv = cv2.calcOpticalFlowFarneback(gray_images[i-1], gray_images[i], None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
            flow_maps.append(flow_cv)
        return flow_maps
    elif method == 'raft' and image_directory and flowmap_directory:
        # https://stackoverflow.com/questions/325463/launch-a-shell-command-with-in-a-python-script-wait-for-the-termination-and-ret
        # calculate pair-wise optical flow maps
        num_digits = 3
        filenames = os.listdir(image_directory)
        for i in range(1, len(filenames)):
            IMAGE_PATH_BEFORE_FRAME = filenames[i-1]
            IMAGE_PATH_AFTER_FRAME = filenames[i]
            SAVE_IMAGE_PATH = os.path.join(flowmap_directory, f"img_{i-1:0>{num_digits}}.png")
            command = f"python3 raft.py -i {IMAGE_PATH_BEFORE_FRAME} {IMAGE_PATH_AFTER_FRAME} -s {SAVE_IMAGE_PATH}"
            process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
            process.wait()

        # read pair-wise optical flow maps
        flow_maps = readImages(flowmap_directory)
        return flow_maps
    else:
        return None


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

    # 2. align images using the first frame as the reference
    aligned_images_directory = "examples/helen/aligned_images"
    images = getAlignedImaged(aligned_images_directory, images)

    # 2. read/calculate optical flow maps
    flow_maps = readImages(flowmap_directory)
    if not flow_maps:
        flow_maps = calculateOpticalFlow(images, method='raft', image_directory=image_directory, flowmap_directory=flowmap_directory)
    print(f"number of flow maps: {len(flow_maps)} = N-1")
    print()

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