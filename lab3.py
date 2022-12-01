import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from PIL import Image


def kmeans_segm(image, K, L, seed=42, spatial=False):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with 
    cluster it is associated to.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """
    # Randomly initialize the K cluster centers
    np.random.seed(seed)
    c = image.shape[2]
    aug_image = image
    if spatial:  # just an experiment, not asked for in the assignment
        X, Y = np.meshgrid(
            np.arange(image.shape[1]), np.arange(image.shape[0]))
        aug_image = np.concatenate(
            (image, X[:, :, np.newaxis], Y[:, :, np.newaxis]), axis=2)
    d = aug_image.shape[2]

    Ivec = np.reshape(aug_image, (-1, d))
    center_idx = np.random.randint(0, Ivec.shape[0], K)
    centers = Ivec[center_idx, :]
    old_segmentation = np.zeros(Ivec.shape[0])
    # Iterate L times
    for i in range(L):
        # Compute all distances between pixels and cluster centers
        distances = distance_matrix(Ivec[:, :c], centers[:, :c]) \
            + 0.1 * distance_matrix(Ivec[:, c:], centers[:, c:])
        # Assign each pixel to the cluster center for which the distance is minimum
        segmentation = np.argmin(distances, axis=1)
        # Recompute each cluster center by taking the mean of all pixels assigned to it
        for k in range(K):
            centers[k] = np.mean(Ivec[segmentation == k], axis=0)

        # Check if the segmentation has converged
        if np.array_equal(segmentation, old_segmentation):
            print('converged after', i+1, 'iterations')
            break
        old_segmentation = segmentation
    segmentation = segmentation.reshape(image.shape[:2])
    return segmentation, centers[:, :c]


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """
    return prob


if __name__ == "__main__":
    # Load the image
    filename_orange = "images-jpg/orange.jpg"
    img_orange = Image.open(filename_orange)
    I_orange = np.asarray(img_orange).astype(np.float32)
    # Display the image
    plt.figure()
    plt.imshow(img_orange)
    plt.title('Original image')
    plt.axis('off')

    # Number of clusters
    K = 4
    # Number of iterations
    L = 25
    segmentation, centers = kmeans_segm(I_orange, K, L)
    # Display the segmentation
    seg_I = centers[segmentation].reshape(I_orange.shape)
    seg_img = Image.fromarray(seg_I.astype(np.uint8))
    plt.figure()
    plt.imshow(seg_img)
    plt.title('Segmentation')
    plt.axis('off')

    if 0:
        segmentation, centers = kmeans_segm(I_orange, K, L, spatial=True)
        # Display the segmentation
        seg_I = centers[segmentation].reshape(I_orange.shape)
        seg_img = Image.fromarray(seg_I.astype(np.uint8))
        plt.figure()
        plt.imshow(seg_img)
        plt.title('Spatial Segmentation')
        plt.axis('off')
        plt.show()
        # Display the cluster centers
