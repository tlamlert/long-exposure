import torch
import cv2

#########################
# The RAFT model that we will use accepts RGB float images with pixel values in
# [-1, 1]. The frames we got from :func:`~torchvision.io.read_video` are int
# images with values in [0, 255], so we will have to pre-process them. We also
# reduce the image sizes for the example to run faster. Image dimension must be
# divisible by 8.

from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
import torchvision.transforms.functional as F

def preprocess(img1_batch, img2_batch, weights=Raft_Small_Weights.DEFAULT):
    transforms = weights.transforms()
    return transforms(img1_batch, img2_batch)


####################################
# Estimating Optical flow using RAFT
# ----------------------------------
# We will use our RAFT implementation from
# :func:`~torchvision.models.optical_flow.raft_large`, which follows the same
# architecture as the one described in the `original paper <https://arxiv.org/abs/2003.12039>`_.
# We also provide the :func:`~torchvision.models.optical_flow.raft_small` model
# builder, which is smaller and faster to run, sacrificing a bit of accuracy.

from torchvision.models.optical_flow import raft_small

def calculateRaftOpticalFlow(images):
    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Raft_Small_Weights.DEFAULT

    # convert images to tensor
    tensor_images = [torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)) for img in images]

    # batch input images
    img1_batch = torch.stack(tensor_images[:-1])
    img2_batch = torch.stack(tensor_images[1:])

    # proprocess
    img1_batch, img2_batch = preprocess(img1_batch, img2_batch, weights=weights)

    # model inference
    model = raft_small(weights=weights, progress=False).to(device)
    model = model.eval()

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flows = [flow.transpose(1, 2, 0) for flow in list_of_flows[-1].detach().numpy()]

    return predicted_flows