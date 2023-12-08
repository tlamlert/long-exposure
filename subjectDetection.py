import head_segmentation.segmentation_pipeline as seg_pipeline
import head_segmentation.visualization as vis
import matplotlib.pyplot as plt
from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
# from pysaliency.plotting import visualize_distribution

import deepgaze_pytorch

VISUALIZE = True

def getHeadSegmentation(image):
    '''
    Credit: https://github.com/wiktorlazarski/head-segmentation
    Returns: A mask of the head of the person in the image
    '''
    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline()
    segmentation_map = segmentation_pipeline.predict(image)
    if VISUALIZE:
        visualizer = vis.VisualizationModule()
        figure, _ = visualizer.visualize_prediction(image, segmentation_map)
        plt.show()
    return segmentation_map

def visualizeAttention(image,log_density_prediction ):
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    axs[0].imshow(image)
    axs[0].set_axis_off()
    # axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
    axs[1].matshow(log_density_prediction) 
    axs[1].set_axis_off()
    # visualize_distribution(log_density_prediction.detach().cpu().numpy()[0, 0], ax=axs[2])
    # axs[2].set_axis_off()
    plt.show()

def getAttentionMask(image):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # you can use DeepGazeI or DeepGazeIIE
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

    # image = face()

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
    # centerbias_template = np.load('centerbias_mit1003.npy')
    centerbias_template = np.zeros((1024, 1024))
    # rescale to match image size
    centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    log_density_prediction = model(image_tensor, centerbias_tensor)
    # print(log_density_prediction.shape)
    # print(np.max(log_density_prediction))
    # print(np.min(log_density_prediction))
    numpy_log_density_prediction = log_density_prediction.detach().numpy()
    mask = numpy_log_density_prediction[0, 0]
    # print(mask.shape)
    # print(mask.shape)
    # print(np.max(mask))
    # print(np.min(mask))
    return mask
    # return np.exp(mask)



def normalize(array):
    # print("YO SHAPE", array.shape)
    return (array-np.min(array))/(np.max(array)-np.min(array))
    


# img0_path = "./examples/helen/img_0.JPG"
# img0_path = "./examples/tiger/img_0.JPG"
# img1_path = "./examples/img_1.JPG"
# img0 = cv2.imread(img0_path, cv2.COLOR_BGR2RGB)
# cv2.imshow("akldsj;",img0)
# attention_mask = getAttentionMask(img0)
# visualizeAttention(img0, attention_mask)
# print("ABOUT TO NORMALIZE")
# attention_mask = normalize(attention_mask)
# visualizeAttention(img0, attention_mask)
# cv2.imshow("attention_mask",attention_mask)
# print("attention_mask min", np.min(attention_mask))
# print("attention_mask max", np.max(attention_mask))

# # img1 = cv2.cvtColor(cv2.imread(img1_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
# head_mask = getHeadSegmentation(img0) #f
# head_mask = normalize(head_mask)
# cv2.imshow("head_mask",head_mask)
# face_mask = attention_mask * (1+head_mask)
# face_mask = normalize(face_mask)
# cv2.imshow("face_mask",face_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("head_mask min", np.min(head_mask))
# print("head_mask max", np.max(head_mask))
# print(face_mask.shape)