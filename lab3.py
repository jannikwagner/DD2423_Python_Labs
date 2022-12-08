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

    # Let I be a set of pixels and V be a set of K Gaussian components in 3D (R,G,B).

    # Store all pixels for which mask=1 in a Nx3 matrix
    mask = mask.astype(bool)
    EPS = 1e-10
    d = image.shape[-1]
    masked_I = image[mask]
    masked_Ivec = np.reshape(masked_I, (-1, d)).astype(np.float32)
    W = np.ones(K) / K

    # Randomly initialize the K components using masked pixels
    center_idx = np.random.randint(0, masked_Ivec.shape[0], K)
    centers = masked_Ivec[center_idx, :]

    # segmentation, centers = kmeans_segm(image, K, L, seed=42, spatial=False)
    print(centers)

    covariances = np.zeros((K, d, d))
    covariances[:] = np.eye(d)  # ugly syntax, but it works

    # Iterate L times
    for i in range(L):
        # Expectation: Compute probabilities P_ik using masked pixels
        G = normpdf(masked_Ivec, centers, covariances, EPS)

        P = W * G
        prob = np.sum(P, axis=1)
        P /= prob[:, np.newaxis] + EPS

        # Maximization: Update weights, means and covariances using masked pixels
        W = np.mean(P, axis=0)
        print(W)
        centers = np.sum(P[:, :, np.newaxis] * masked_Ivec[:, np.newaxis, :],
                         axis=0) / (np.sum(P, axis=0)[:, np.newaxis] + EPS)
        print(centers)
        for k in range(K):
            centered = masked_Ivec - centers[k]
            covariances[k] = np.sum(P[:, k, np.newaxis, np.newaxis] * (
                centered[:, np.newaxis, :] * centered[:, :, np.newaxis]), axis=0) / (np.sum(P[:, k]) + EPS)
        print(covariances)

    # Compute probabilities p(c_i) in Eq.(3) for all pixels I.
    Ivec = np.reshape(image, (-1, d)).astype(np.float32)
    G = normpdf(Ivec, centers, covariances, EPS)
    P = W * G
    prob = np.sum(P, axis=1)
    return prob


def normpdf(Ivec, centers, covariances, EPS=1e-10):
    K, d = centers.shape
    G = np.zeros((Ivec.shape[0], K))
    for k in range(K):
        print(k)
        centered = Ivec - centers[k]
        G[:, k] = np.sum(
            centered * (np.linalg.inv(covariances[k]) @ centered.T).T, axis=1)
    G = np.exp(-0.5 * G)
    G /= np.sqrt(np.linalg.det(covariances)*(2*np.pi)**d) + EPS
    return G


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
    K = 6
    # Number of iterations
    L = 25
    mask = np.ones(I_orange.shape[:2]).astype(bool)
    mixture_prob(I_orange, K, L, mask)
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
