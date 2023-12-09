import os
import cv2
import numpy as np
import torch
from torchvision.utils import flow_to_image

# internal imports
from alignImages import alignImages
from composite import calc_Mflow, alpha_blending
from raft import calculateRaftOpticalFlow
from subjectDetection import getAttentionMask, getHeadSegmentation, normalize
from drawMask import getMask

"""
Long Exposure Pipeline

1. read all images

2. align images using the first frame as the reference

3. subject detection (creating the face mask) -> one face mask

4. interpolate between frames -> one blurred image

5. composite
"""


def readImages(directory, resize_scale=1):
    """
    Read and return a list of images from directory

    Returns:
        List of images
    """
    images = []
    for filename in sorted(os.listdir(directory)):
        if os.path.isdir(filename) or filename == "aligned_images" or filename == "output" or filename == "flow_map":
            continue
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_COLOR)
        if resize_scale != 1:
            img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        images.append(img)
    return images


def writeImages(directory, images):
    """
    Write images to directory
    """
    num_digits = 3
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(directory, f"img_{i:0>{num_digits}}.png"), img)


def getAlignedImaged(images, from_cache=False, directory=None):
    # Read in cached aligned iimages if exists
    if from_cache and directory and len(os.listdir(directory)) > 0:
        alignedImages = []
        for filename in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, filename))
            alignedImages.append(img)
        return alignedImages

    # Else generate aligned images and save to directory
    alignedImages = alignImages(images)
    if directory:
        writeImages(directory, alignedImages)
        
    return alignedImages


def naiveBlurImages(images):
    # naively blur all images together by taking the average
    return np.mean(np.stack(images, -1), axis=-1)


def calculateOpticalFlow(images, method, from_cache=False, flowmap_dir=None):
    if flowmap_dir:
        flowmap_dir = os.path.join(flowmap_dir, method)
        os.makedirs(flowmap_dir, exist_ok=True)
    
    # load from cache if possible
    if from_cache and os.path.exists(flowmap_dir) and len(os.listdir(flowmap_dir)) > 0:
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


def subjectDetection(image, face_enable=False):
    """ 
    From an image, produces and returns a face mask

    Returns:
        Face mask
    """
    # TODO: Maybe also allow user to use their own custom mask

    # Find gaze attention mask
    print("Finding gaze attention mask...")
    attention_mask = getAttentionMask(image) #s
    attention_mask = normalize(attention_mask)
    if not face_enable:
        return attention_mask
    
    # Find subject mask
    print("Finding head mask...")
    head_mask = getHeadSegmentation(image) #f
    head_mask = normalize(head_mask)
    
    # Combine both of them according to equation in paper
    face_mask = attention_mask * (1+head_mask)
    face_mask = normalize(face_mask)
    # cv2.imshow("face_mask",face_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return face_mask


def interpolateFrames(frame1, frame2, input_flow_map, num_frames):
    """
    Takes in 2 sequential frames, produces a list of frames in between
    
    Returns:
        List of inbetween frames
    """

    def generateOneFrame(flow_map, t):
        h, w = flow_map.shape[:2]

        # Forward pass
        flow_map = flow_map * t
        flow_map[:,:,0] += np.arange(w)
        flow_map[:,:,1] += np.arange(h)[:,np.newaxis]

        # Backward pass
        return cv2.remap(frame1, flow_map, None, cv2.INTER_LINEAR)

    inbetween_frames = [generateOneFrame(input_flow_map, t/num_frames) for t in range(1, num_frames)]
    return inbetween_frames


def blurImages(images, flow_maps):
    """
    Takes a series of images and blurs them together via linear interpolation.
    
    Returns:
        One blurred image
    """
    blurred_image = images[0]
    weight = 1
    for i in range(1, len(images)):
        # generate inbetween
        NUM_FRAMES = 1 << 4 - 1
        print("i: ", i)
        print("images[i].shape: ", images[i].shape) 
        print("images[i-1].shape: ", images[i-1].shape)
        print("flow_maps[i-1].shape: ", flow_maps[i-1].shape)
        inbetween_frames = interpolateFrames(images[i-1], images[i], flow_maps[i-1], NUM_FRAMES)
        inbetween_frames.append(images[i])
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
    print("np.max(MFlow) BEFORE", np.max(MFlow))
    MFlow = normalize(MFlow)
    print("np.max(MFlow) After", np.max(MFlow))

    # combine the flow and the clipped face masks with a simple max operator
    flow_face_mask = np.where(MFlow > subject_mask, MFlow, subject_mask)
    cv2.imshow("flow_face_mask", flow_face_mask) 
    cv2.waitKey(0)
    cv2.imshow("sharp_image", sharp_image) 
    cv2.waitKey(0)
    cv2.imshow("blurred_image", blurred_image) 
    cv2.waitKey(0)
    blurred_image = blurred_image
    cv2.imshow("blurred_image_after", blurred_image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    print("np.max(sharp_image)", np.max(sharp_image))
    print("np.max(flow_face_mask) BEFORE", np.max(flow_face_mask))
    flow_face_mask = normalize(flow_face_mask)
    print("np.max(flow_face_mask) After", np.max(flow_face_mask))

    # flow_face_mask = np.zeros_like(subject_mask)
    
    # use composite function(s) from previous previous project code
    composite = alpha_blending(sharp_image, flow_face_mask, blurred_image)

    return composite
     

def pipeline():
    # 0. prepare directories
    image_directory = "examples/helen_face_mov"
    flowmap_directory = os.path.join(image_directory, "flow_map")
    aligned_images_directory = os.path.join(image_directory, "aligned_images")
    output_directory = os.path.join(image_directory, "output")
    os.makedirs(flowmap_directory, exist_ok=True)
    os.makedirs(aligned_images_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    # 1.1. read all images
    print("Reading Images...")
    images = readImages(image_directory, resize_scale=1/4)
    images = images[:8]
    print(f"number of images: {len(images)} = N")
    print(f"image shape: {images[0].shape} = (H, W, 3)")
    print()

    method = "raft"
    # 1.1.1. resize images(if raft)
    if method == "raft":
        H, W = images[0].shape[:2]
        H -= H % 8
        W -= W % 8
        images = [cv2.resize(img, (W, H)) for img in images]

    # 1.2. align images using the first frame as the reference
    print("Aligning Images...")
    images = getAlignedImaged(images, from_cache=False, directory=aligned_images_directory)
    print(f"number of aligned images: {len(images)} = N")
    print(f"aligned image shape FIRST: {images[0].shape} = (H, W, 3)")
    print(f"aligned image shape LAST: {images[-1].shape} = (H, W, 3)")

    # 1.3. naive long exposure
    print("Naively blurring images...")
    naive_blurred = naiveBlurImages(images)
    cv2.imwrite(os.path.join(output_directory, "naive_blurred.png"), naive_blurred)
    print()

    # 2. read/calculate optical flow maps
    print("Calculating optical flow maps...")
    flow_maps = calculateOpticalFlow(images, method=method, from_cache=False, flowmap_dir=flowmap_directory)
    print(f"number of flow maps: {len(flow_maps)} = N-1")
    print(f"flow shape: {flow_maps[0].shape} = (H, W, 2)")
    print()

    # flow_img = flow_to_image(torch.tensor(flow_maps[0].transpose(2, 0, 1), dtype=torch.float32)).numpy().transpose(1, 2, 0)
    # cv2.imshow("example flow map", flow_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # 3. subject detection
    print("Creating face mask...")
    sharp_image = images[0]
    subject_mask = None
    if True:
        # using first image/sharp image also as base image
        subject_mask = getMask(sharp_image) 
    else:
        face_mask = subjectDetection(sharp_image)
        subject_mask = face_mask
    cv2.imwrite(os.path.join(output_directory, "face_mask.png"), subject_mask * 255)
    print()

    # 4. interpolate between frames -> one blurred image
    print("Linearly interpolating between frames...")
    blurred_image = blurImages(images, flow_maps)
    cv2.imwrite(os.path.join(output_directory, "blurred_image.png"), blurred_image)
    print()

    # 5. composite
    print("Compositing...")
    result = composite(sharp_image, blurred_image, flow_maps, subject_mask)
    cv2.imwrite(os.path.join(output_directory, "result.png"), result)
    print("Finished!")


if __name__ == "__main__":
    pipeline()