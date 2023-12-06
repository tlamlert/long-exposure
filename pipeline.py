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
    # TODO
    pass
    
def alignImages(images):
    """ 
    Taking a series of images, align all images based on the first image 
    
    Returns:
        List of aligned images
    """
    
    alignedImages = None
    return alignedImages

def subjectDetection(images):
    """ From an image, produces and ret a face mask """
    face_mask = None
    return face_mask

def interpolateFrames(frame1, frame2):
    """
    Takes in 2 sequential frames, produces a list of frames in between
    
    Returns:
        List of inbetween frames
    """
    return None

def blurImages(images):
    blurred_image = None
    return blurred_image

def composite(images):
     """
    Composite between MFlow, subject mask, sharp, and blurred to return final output image 
    
    Returns:
        Final output image
    """
     
     