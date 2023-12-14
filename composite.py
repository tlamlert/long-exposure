import cv2
import numpy as np
import os
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

# CONSTANTS AS DEFINED BY THE PAPER
ALPHA = 0.16
BETA = 0.32

def calc_F_ref(F):
    robust_max = np.max(F)
    F_ref = np.zeros_like(F)
    F_ref.fill(robust_max)
    return F_ref


def calc_F(optical_flows):
    # optical_flows shape: N * W * H * 2
    # assuming the optical flows are stacked in the way that the last channel = # optical flows = N.
    # output should be of size W * H
    # (3, 1008, 756, 2)
    mag_list = []
    for optical_f in optical_flows:
        norm = np.linalg.norm(optical_f, axis=-1)
        print("NORM SHAPE", norm.shape)
        mag_list.append(norm)

    optical_flows_array = np.asarray(mag_list)
    print(optical_flows_array.shape)
    result = np.max(optical_flows_array, axis=0)
    print(result.shape)
    return result


def calc_Mflow(optical_flows, sharp_image):
    F = calc_F(optical_flows)
    F_ref = calc_F_ref(F)
    mFlow_numerator = F - np.clip(ALPHA * F_ref, 0, None)
    mFlow_denom = np.clip(BETA * F_ref, None, 1) - np.clip(ALPHA * F_ref, 0, None)
    mFlow = mFlow_numerator / mFlow_denom
    # mFlow = 1 - mFlow
    # mFlow = mFlow.transpose(1, 2, 0)
    
    print("mFlow shape : ", mFlow.shape)
    print("F shape : ", F.shape)
    print("F_ref shape : ", F_ref.shape)
    
    # apply bilateral blur using sharp image as a guide
    bilateral = cv2.bilateralFilter(mFlow, 15, 75, 75)  # -> this is not with sharp image as a guide. Note: if it looks bad, comment this out.
    return bilateral


def alpha_blending(source, mask, target):
    """
    Performs alpha blending. 
    Source, mask, and target are all numpy arrays of the same shape 
    (this is ensured by the fix_images function called in main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask. Could also be matte.
        target - np.array of target image

    Returns:
        np.array of blended image
    """

    return (source * mask[..., None]) + (target * (1 - mask[..., None]))



# ---------------------------------------------------------------------------
def build_laplacian_pyramid_recursive(img, depth, detail_list):
    MAX_DEPTH = 4
    if depth>=MAX_DEPTH: # reached hardcoded low-res limit or max depth
        detail_list.append(img)
        return detail_list
    # blur image
    blur= cv2.GaussianBlur(img,(3,3),0)
    detail = cv2.subtract(img, blur)
    detail_list.append(detail)
    image = cv2.resize(blur, (0, 0), fx=0.5, fy=0.5)
    return build_laplacian_pyramid_recursive(image, depth+1, detail_list)

def build_gaussian_pyramid_recursive(img, depth, blur_list):
    MAX_DEPTH = 4
    #  or b.shape[0]*b.shape[1] < MIN_SIZE_PYRAMID_IMG
    if depth>=MAX_DEPTH: # reached hardcoded low-res limit or max depth
        blur_list.append(img)
        return blur_list
    # blur image
    blur= cv2.GaussianBlur(img,(3,3),0)
    image = cv2.resize(blur, (0, 0), fx=0.5, fy=0.5)
    blur_list.append(blur)
    return build_gaussian_pyramid_recursive(image, depth+1, blur_list)

def reconstruct_laplacian_pyramid(Ls):
    image = Ls[-1]
    for i in range(Ls.__len__() - 2, -1, -1):
        image = cv2.resize(image, (Ls[i].shape[1], Ls[i].shape[0]))
        image = image + Ls[i]
    return image

def laplacian_pyramid_blend(source, mask, target):
    """
    Performs Laplacian pyramid blending (from lab 'compositing'). 
    Source, mask, and target are all numpy arrays of the same shape 
    (this is ensured by the fix_images function called in main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask
        target - np.array of target image

    Returns:
        np.array of blended image
    """
    l1 = build_laplacian_pyramid_recursive(source, 1, [])
    l2 = build_laplacian_pyramid_recursive(target, 1, [])
    gm = build_gaussian_pyramid_recursive(mask, 1, [])
    l_out = []
    for i in range(4):
        l_out.append((gm[i][:,:,None] * l1[i]) + ((1 - gm[i])[:,:,None] * l2[i]))
    output = reconstruct_laplacian_pyramid(l_out)
    return output


def bounds_check_b(b, oneDimIndex, numRows, numCols, col, row, source):
    source_grad = 4*source[row, col]
    if  row+1 < numRows:
        source_grad -= source[row+1, col]

    if  row-1 >= 0:
        source_grad -= source[row-1, col]

    if  col+1< numCols:
        source_grad -= source[row, col+1]
    
    if  col-1 >= 0:
        source_grad -= source[row, col-1]
    return source_grad



def bounds_check_A(A, oneDimIndex, numRows, numCols, col, row, source):
    if row+1 < numRows:
        A[oneDimIndex, (row+1)*source.shape[1] + col] = -1

    if  row-1 >= 0:
        A[oneDimIndex, (row-1)*source.shape[1] + col] = -1

    if  col+1 < numCols:
        A[oneDimIndex, row*source.shape[1] + col+1] = -1
    
    if  col-1 >= 0:
        A[oneDimIndex, row*source.shape[1] + col-1] = -1
    A[oneDimIndex, oneDimIndex] = 4

def poisson_blend_channel(source, mask, target, A, makeA, isAlpha):
    numRows, numCols = source.shape[0], source.shape[1]
    if makeA:
        A = identity(source.shape[0] * source.shape[1]).tolil()
    b = np.zeros(source.shape[0] * source.shape[1])
    for col in range(source.shape[1]):
        for row in range(source.shape[0]):
            # oneDimIndex = col*source.shape[0] + row # acc to unfolding from lecture slides
            oneDimIndex = row*source.shape[1] + col # acc to unfolding from lecture slides
            if mask[row, col] == 0:
                # not under the mask
                b[oneDimIndex] = target[row,col]
            else:
                # under the mask - QUESTION SHOULD I PAD THE SOURCE? - note: all masks have zeroes at the edges so this should be fine
                if makeA:
                    bounds_check_A(A, oneDimIndex, numRows, numCols, col, row, source)
                source_grad = bounds_check_b(b, oneDimIndex, numRows, numCols, col, row, source)
                if isAlpha:
                    target_grad = bounds_check_b(b, oneDimIndex, numRows, numCols, col, row, target)
                    b[oneDimIndex] = max(source_grad, target_grad)
                else:
                    b[oneDimIndex] = source_grad

    A = A.tocsr()
    x = spsolve(A, b)
    # x = x.reshape((source.shape[1], source.shape[0]))
    x = x.reshape(source.shape)
    print("DONE")
    return x, A


# mixing with alpha
def poisson_blend(source, mask, target):
    """
    Performs Poisson blending. Source, mask, and target are all numpy arrays
    of the same shape (this is ensured by the fix_images function called in
    main.py).

    Args:
        source - np.array of source image
        mask   - np.array of binary mask
        target - np.array of target image

    Returns:
        np.array of blended image
    """

    # TODO: Implement this function!
    '''
    We will set up a system of equations A * x = b, where A has as many rows
and columns as there are pixels in our images. Thus, a 300x200 image will
lead to A being 60000 x 60000. 'x' is our output image (a single color
channel of it) stretched out as a vector. 'b' contains two types of known
values:

    (1) For rows of A which correspond to pixels that are not under the
        mask, b will simply contain the already known value from 'target'
        and the row of A will be a row of an identity matrix. Basically,
        this is our system of equations saying "do nothing for the pixels we
        already know".
    (2) For rows of A which correspond to pixels under the mask, we will
        specify that the gradient (actually the discrete Laplacian) in the
        output should equal the gradient in 'source', according to the final
        equation in the webpage:
           4*x(i,j) - x(i-1, j) - x(i+1, j) - x(i, j-1) - x(i, j+1) =
           4*s(i,j) - s(i-1, j) - s(i+1, j) - s(i, j-1) - s(i, j+1)
        The right hand side are measurements from the source image. The left
        hand side relates different (mostly) unknown pixels in the output
        image. At a high level, for these rows in our system of equations we
        are saying "For this pixel, I don't know its value, but I know that
        its value relative to its neighbors should be the same as it was in
        the source image".
    '''
    # x = np.zeros((source.shape[1], source.shape[0],3))
    x = np.zeros(source.shape)
    x[:,:,0], A = poisson_blend_channel(source[:,:,0], mask[:,:,0], target[:,:,0], None, True, False)
    x[:,:,1], _ = poisson_blend_channel(source[:,:,1], mask[:,:,0], target[:,:,1], A, False, False)
    x[:,:,2], _ = poisson_blend_channel(source[:,:,2], mask[:,:,0], target[:,:,2], A, False, False)
    return x

def composite(source_dir, sharp_image_idx, mask_dir, target_dir):
    curr_idx = 0
    sharp_image = None
    for filename in sorted(os.listdir(source_dir)):
        if curr_idx == sharp_image_idx:
            sharp_image = cv2.imread(os.path.join(source_dir, filename), cv2.IMREAD_COLOR)
        curr_idx+=1
    assert sharp_image is not None
    # sharp_image = cv2.resize(sharp_image, (0, 0), fx=1/4, fy=1/4)
    # if True: # method == "raft"
    #     H, W = sharp_image.shape[:2]
    #     H -= H % 8
    #     W -= W % 8
    #     sharp_image = cv2.resize(sharp_image, (W, H))
    print(sharp_image.shape)
    mask = cv2.imread(os.path.join(mask_dir, "flow_face_mask.png"), cv2.IMREAD_GRAYSCALE)
    blur  = cv2.imread(os.path.join(target_dir, "blurred_image.png"), cv2.IMREAD_COLOR)
    # BGR -> RGB:
    source = sharp_image[..., ::-1]
    target = blur[..., ::-1]

    # uint8 -> float32
    source = source.astype(np.float32) / 255.
    target = target.astype(np.float32) / 255.
    mask = np.round(mask.astype(np.float32) / 255.)
    mask_for_composite = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask_for_composite[:,:,0] = mask
    mask_for_composite[:,:,1] = mask
    mask_for_composite[:,:,2] = mask
    print(blur.shape)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.imshow("sharp_image", sharp_image)
    cv2.waitKey(0)
    cv2.imshow("blur", blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # result = poisson_blend(sharp_image, mask_for_composite, blur)
    print(np.max(sharp_image))
    print(np.max(mask))
    print(np.max(blur))
    result = alpha_blending(source, mask, target)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    result = poisson_blend(source, mask_for_composite, target)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(target_dir, "result_composite.png"), result[..., ::-1]*255)


# image_directory = "examples/hair_close/"
# output_directory = os.path.join(image_directory, "output")
# source_directory = os.path.join(image_directory, "aligned_images")
# composite(source_directory, 5, output_directory, output_directory)