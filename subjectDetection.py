import head_segmentation.segmentation_pipeline as seg_pipeline
import head_segmentation.visualization as vis
import matplotlib.pyplot as plt
from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import matplotlib.pyplot as plt
from pysaliency.plotting import visualize_distribution

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
    axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
    axs[1].set_axis_off()
    visualize_distribution(log_density_prediction.detach().cpu().numpy()[0, 0], ax=axs[2])
    axs[2].set_axis_off()
    plt.show()


def getAttentionMask(image):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # you can use DeepGazeI or DeepGazeIIE
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

    image = face()

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
    centerbias_template = np.load('centerbias_mit1003.npy')
    # rescale to match image size
    centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    log_density_prediction = model(image_tensor, centerbias_tensor)
