import cv2
import numpy as np

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
    mFlow_numerator = F - ALPHA * F_ref
    mFlow_denom = BETA * F_ref - ALPHA * F_ref
    mFlow = mFlow_numerator / mFlow_denom
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