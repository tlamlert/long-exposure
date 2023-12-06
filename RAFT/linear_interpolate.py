'''
Synthetic Shutter Speed Imaging
https://users.soe.ucsc.edu/~prabath/telleenj_SSSI_sub_final.pdf

Controlling Motion Blur in Synthetic Long Time Exposures
https://cgl.ethz.ch/Downloads/Publications/Papers/2019/Lan19a/Lan19a.pdf

IDK what this is
https://link.springer.com/chapter/10.1007/978-3-540-24673-2_3
'''

import cv2
import numpy as np

def visualizeFlow(flow, name="colored flow"):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow(name, bgr)
    return bgr


# read images
img0_path = "../examples/img_0.JPG"
img1_path = "../examples/img_1.JPG"
img0 = cv2.imread(img0_path, cv2.IMREAD_UNCHANGED)
img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
img0 = cv2.resize(img0, (960, 520))
img1 = cv2.resize(img1, (960, 520))
cv2.imshow("img0", img0)

# compute flow using cv2
img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
flow_cv = cv2.calcOpticalFlowFarneback(img0_gray, img1_gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
flow_cv_back = cv2.calcOpticalFlowFarneback(img1_gray, img0_gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
print(f"flow shape = {flow_cv.shape}") # (H, W, 2)
print(f"min = {flow_cv.min()}, max = {flow_cv.max()}")
visualizeFlow(flow_cv, "cv2 flow")

# read flow map
predicted_flows = np.load("output/predicted_flows_forward.npy")
print("predicted_flows", predicted_flows.shape)
flow_cv = predicted_flows[0, ...].transpose(1, 2, 0)

predicted_flows = np.load("output/predicted_flows_backward.npy")
print("predicted_flows", predicted_flows.shape)
flow_cv_back = predicted_flows[0, ...].transpose(1, 2, 0)

# Overlap img0 w/ the flow m
# test = 0.5 * img0 + 0.5 * flow_pic
# cv2.imshow("combined", test.astype(np.uint8))
# print(f"dtype = {predicted_flows.dtype}")
# print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
# print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

# print("transposed shape", flow.shape)
# print("image 0", flow.shape)


def generateFrame(img0, flow_cv, t, flow_cv_back, img1):
    # Forward pass
    flow_cv = flow_cv * t
    print(t)
    h, w = flow_cv.shape[:2]
    flow_cv[:,:,0] += np.arange(w)
    flow_cv[:,:,1] += np.arange(h)[:,np.newaxis]
    new_frame = cv2.remap(img0, flow_cv, None, cv2.INTER_LINEAR)
    return new_frame

    # Backward pass
    flow_cv_back = flow_cv_back * (1.0-t)
    flow_cv_back[:,:,0] += np.arange(w)
    flow_cv_back[:,:,1] += np.arange(h)[:,np.newaxis]
    new_frame_back = cv2.remap(img1, flow_cv_back, None, cv2.INTER_LINEAR)
    cv2.imshow("merged", (0.5*new_frame + 0.5*new_frame_back).astype(np.uint8))
    return new_frame_back

    return 0.5*new_frame + 0.5*new_frame_back


num_frames = 100
inbetween_frames = [img0] + [generateFrame(img0, flow_cv, t/num_frames, flow_cv_back, img1) for t in range(1, num_frames)] + [img1]
blurred = np.mean(np.stack(inbetween_frames, axis=-1), axis=-1).astype(np.int32)
cv2.imwrite("blurred.png", blurred)

for i, frame in enumerate(inbetween_frames):
    cv2.imwrite(f"helen/img{i}.png", frame)


cv2.waitKey(0)
cv2.destroyAllWindows()
# # generate one new frame
# new_frame = generateFrame(img0, predicted_flows[0, ...], 1)
# print(f"new_frame shape = {new_frame.shape} = (H, W, 3)")
# cv2.imwrite("output/generated_frame.png", new_frame)

# # 1. compute number of inbetween frames needed (largest pixel motion across all frames)
# num_frames = compute something baby

# # 2. generate inbetween frames using linear interpolation
# inbetween_frames = []

# # 3. average all generated inbetween frames to get the blurred result
# blurred = np.mean(np.stack(inbetween_frames, axis=-1), axis=-1).astype(np.int32)
# cv2.imwrite("blurred.png", blurred)