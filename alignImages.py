import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from skimage.transform import AffineTransform

# ===============================================================
#                      Aligning Images
# ===============================================================
def alignImages(images):
    """ 
    Taking a series of images, align all images based on the first image 
    
    Returns:
        List of aligned images
    """
    aligned_images = []
    root_img = images[0]
    aligned_images.append(root_img)

    for img in images[1:]:
        # Find correspondences between root_img and img
        pointsRoot, pointsImg = find_correspondences(root_img, img)

        # Compute homography transformation that transforms img -> root_img
        mat = calculate_transform(pointsImg, pointsRoot, type='rigid')

        # Warp img
        warpedImg = warp_images(img, root_img, mat)
        # warpedImg = warp_images(root_img, img, mat)

        aligned_images.append(warpedImg)

    return aligned_images


def calculate_transform(pointsA, pointsB, type='homography'):
    """Returns the matrix that aligns pointsA to pointsB"""
    if type == 'rigid':
        rigid, _ = cv2.estimateAffinePartial2D(pointsA, pointsB, False)
        mat = np.array([[rigid[0][0], rigid[0][1], rigid[0][2]],
                        [rigid[1][0], rigid[1][1], rigid[1][2]],
                        [0, 0, 1]])
    elif type == 'homography':
        mat, _ = cv2.findHomography(pointsA, pointsB, cv2.USAC_MAGSAC, 5.0)
    return mat


def warp_images(A, B, transform_M):
    """Warps B to match A. Returns warped_B"""
    # Step 1 - Find the bounding box of transformed/warped A in the coordinate frame of B
    #   so that we can determine the dimensions of our composited image.
    A_h, A_w = A.shape[1], A.shape[0]
    A_rect = np.array([[0, 0], [0, A_w-1], [A_h-1, 0], [A_h-1, A_w-1]], dtype=np.float32) # Coordinates (row, col) of the rectangle defining A (top-left, top-right, bottom-left, bottom-right)
    warped_A_rect = cv2.perspectiveTransform(A_rect[np.newaxis], transform_M) 
    warped_A_rect = warped_A_rect[0]

    # Step 2 - Calculate the translation, if any, that is needed to bring A into fully nonnegative coordinates. 
    #   If we transform A without regard to the bounds, it may get cropped. 
    dRow = -min(warped_A_rect[0][0], warped_A_rect[1][0], warped_A_rect[2][0], warped_A_rect[3][0], 0) 
    dCol = -min(warped_A_rect[0][1], warped_A_rect[1][1], warped_A_rect[2][1], warped_A_rect[3][1], 0) 
    translation_xy = AffineTransform(translation=(dRow, dCol))

    # Step 3 - Calculate the width and height of the output image.
    B_h, B_w = B.shape[1], B.shape[0]
    H = A.shape[1]
    W = A.shape[0]
    # H = dRow + max(B_h-1, warped_A_rect[0][0], warped_A_rect[1][0], warped_A_rect[2][0], warped_A_rect[3][0])
    # W = dCol + max(B_w-1, warped_A_rect[0][1], warped_A_rect[1][1], warped_A_rect[2][1], warped_A_rect[3][1])

    # Create a translation transform T that translates B to account for any shift of A. This is a 2x3 affine matrix representing the translation.
    transform_T = np.array(translation_xy, dtype=np.float32)[:2, :]

    # Update transform M with the translation needed to keep A in frame.
    # transform_M = np.concatenate((transform_T, [[0, 0, 1]]), axis=0) @ transform_M

    # Create the warped images
    # warped_A = cv2.warpPerspective(A, transform_M, (int(H),int(W)))
    warped_B = cv2.warpAffine(B, transform_T, (int(H),int(W)))

    return warped_B


# ===============================================================
#      FINDING FEATURES / CORRESPONDENCES BETWEEN TWO IMAGES
# ===============================================================

def find_keypoints_and_features(img):
    '''
    input:
        img:            input image
        block_size:     parameter to define block_size x block_size neighborhood around 
                        each pixel used in deciding whether it's a corner

    returns:        
        keypoints:      an array of xy coordinates of interest points
        features:       an array of features corresponding to each of the keypoints
    '''

    image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') # this is an alternate 8-bit conversion from the grayscale conversion that also normalizes the image. seems to work better with sift. found on stack overflow https://stackoverflow.com/questions/50298329/error-5-image-is-empty-or-has-incorrect-depth-cv-8u-in-function-cvsift

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image8bit,None)

    kp = [[point.pt[0], point.pt[1]] for point in kp]

    return kp, des

def find_correspondences(imgA, imgB):
    ''' 
    Automatically computes a set of correspondence points between two images.

    imgA:         input image A
    imgB:         input image B
    block_size:   size of the area around an interest point that we will 
                  use to create a feature vector. Default to 

    pointsA:      xy locations of the correspondence points in image A
    pointsB:      xy locations of the correspondence points in image B   
    '''

    # Step 1:   Use Harris Corner Detector to find a list of interest points in both images, 
    #   and compute features descriptors for each of those keypoints. Here, we are calculating
    #   the robust SIFT (i.e. Scale Invariant Feature Transform) descriptors, which detects corners
    #   at multiple scales.

    kp1, des1 = find_keypoints_and_features(imgA)
    kp2, des2 = find_keypoints_and_features(imgB)

    # Step 2: Find correspondences between the interest points in both images using the feature
    #   descriptors we've calculated for each of the points. 
    #
    # - Step 2a: Calculate and store the distance between feature vectors of all pairs (one from A and one from B) 
    #   of interest points. 
    #   As you may recall, there are many possible distance/similarity metrics. You're welcome to experiment 
    #   but we recommend the L2 norm, tried and true. (hint: scipy.spatial.distance.cdist)

    distances = scipy.spatial.distance.cdist(des1, des2) # returns a matrix such that at the distance between i, j is at index i, j

    # - Step 2b: Find the best matches (pairs of points with the most similarity) that are below some error threshold. 
    #   You're aiming for some number of matches greater than MIN_NUMBER_MATCHES, otherwise you may not have enough information
    #   for later steps. Each point should only have one match, and we want to throw out any points that have no matches.

    MIN_NUMBER_MATCHES = 20
    FEATURE_THRESHOLD = 0.2

    # sort the distances. smaller is better
    sorted_distances_indices = np.argsort(distances) #returns the indices that would sort distances
    sorted_distances = np.take_along_axis(distances, sorted_distances_indices, -1)

    # the top two best matches in a row are compared <-- actually two best matches in a column
    # we want this for the ratio test https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    top_two_match_ratios = sorted_distances[:,0] / sorted_distances[:, 1]

    # sort the comparisons
    sorted_ratio_indices = np.argsort(top_two_match_ratios)
    sorted_ratios = top_two_match_ratios[sorted_ratio_indices]

    # find the indices of A's keypoints that produce the smallest match distance ratios
    bestA = sorted_ratio_indices[sorted_ratios < FEATURE_THRESHOLD]
    if len(bestA) < MIN_NUMBER_MATCHES:
        #if we don't have enough matches below the threshold, take the top MIN_NUMBER_MATCHES
        bestA = sorted_ratio_indices[:MIN_NUMBER_MATCHES]
    
    # get best B from best A
    bestB = sorted_distances_indices[bestA, 0]

    pointsA = np.array(kp1)[bestA]
    pointsB = np.array(kp2)[bestB]

    return pointsA, pointsB


def visualizeCorrespondences(imgA, imgB, pointsA, pointsB):
    """ Visualizes the correspondences between imgA and imgB """
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(imgA)
    ax2.imshow(imgB)

    for xy1, xy2 in zip(pointsA, pointsB):
        con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color="red")
        ax2.add_artist(con)
        ax1.plot(xy1[0], xy1[1], 'ro', markersize=2)
        ax2.plot(xy2[0], xy2[1], 'ro', markersize=2)

    plt.show()


