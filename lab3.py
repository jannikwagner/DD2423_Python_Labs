import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from PIL import Image
from scipy import stats


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
    c = image.shape[-1]
    aug_image = image
    if spatial:  # just an experiment, not asked for in the assignment
        X, Y = np.meshgrid(
            np.arange(image.shape[1]), np.arange(image.shape[0]))
        aug_image = np.concatenate(
            (image, X[:, :, np.newaxis], Y[:, :, np.newaxis]), axis=2)
    d = aug_image.shape[-1]

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
    segmentation = segmentation.reshape(image.shape[:-1])
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
    EPS = 1e-6
    d = image.shape[-1]
    masked_I = image[mask]
    masked_Ivec = np.reshape(masked_I, (-1, d)).astype(np.float32)

    INIT_KMEANS = True
    if not INIT_KMEANS:
        # Randomly initialize the K components using masked pixels
        center_idx = np.random.randint(0, masked_Ivec.shape[0], K)
        centers = masked_Ivec[center_idx, :]
        W = np.ones(K) / K
    else:
        # Initialize the K components using K-means on masked pixels
        segmentation, centers = kmeans_segm(
            masked_Ivec, K, L, seed=42, spatial=False)
        W = segmentation[:, np.newaxis] == np.arange(K)[np.newaxis, :]
        W = W.sum(axis=0) / W.sum()
    # print("W:", W)
    # print("centers:", centers)

    covariances = np.zeros((K, d, d))
    covariances[:] = np.eye(d)*100.  # ugly syntax, but it works
    # print("covariances:", covariances)

    # Iterate L times
    for i in range(L):
        # print("i:", i)
        # Expectation: Compute probabilities P_ik using masked pixels
        G = normpdf2(masked_Ivec, centers, covariances)
        # print("G:", G)

        P = W * G
        prob = np.sum(P, axis=1)
        P = P / (prob[:, np.newaxis] + EPS)
        # print("P", P)

        # Maximization: Update weights, means and covariances using masked pixels
        W = np.mean(P, axis=0)
        # print("W:", W)
        centers = np.sum(P[:, :, np.newaxis] * masked_Ivec[:, np.newaxis, :],
                         axis=0) / (np.sum(P, axis=0)[:, np.newaxis] + EPS)
        # print("centers:", centers)

        for k in range(K):
            centered = masked_Ivec - centers[k]
            covariances[k] = np.sum(P[:, k, np.newaxis, np.newaxis] * (
                centered[:, :, np.newaxis] @ centered[:, np.newaxis, :]), axis=0) / (np.sum(P[:, k]) + EPS)
        # print("covariances:", covariances)

    # Compute probabilities p(c_i) in Eq.(3) for all pixels I.
    Ivec = np.reshape(image, (-1, d)).astype(np.float32)
    G = normpdf2(Ivec, centers, covariances)
    P = W * G
    prob = np.sum(P, axis=1)
    return prob


def normpdf(Ivec, centers, covariances, EPS=1e-10):
    K, d = centers.shape
    G = np.zeros((Ivec.shape[0], K))
    for k in range(K):
        # print(k)
        centered = Ivec - centers[k]

        temp = np.sum(covariances[np.newaxis, k, :, :]
                      * centered[:, :, np.newaxis], axis=1)
        temp = np.sum(temp * centered, axis=-1)

        temp2 = np.sum(
            centered * (np.linalg.inv(covariances[k]) @ centered.T).T, axis=1)

        temp3 = centered[:, np.newaxis,
                         :] @ np.linalg.inv(covariances[k]) @ centered[:, :, np.newaxis]
        temp3 = temp3.reshape(-1)
        print(np.sum((temp - temp2)**2))
        print(np.sum((temp - temp3)**2))
        print(np.sum((temp2 - temp3)**2))

        G[:, k] = temp3
    G = np.exp(-0.5 * G)
    G /= np.sqrt(np.linalg.det(covariances)*(2*np.pi)**d) + EPS
    return G


def normpdf2(Ivec, centers, covariances):
    K, d = centers.shape
    G = np.zeros((Ivec.shape[0], K))
    for k in range(K):
        # print(k)
        G[:, k] = stats.multivariate_normal.pdf(
            Ivec, centers[k], covariances[k])
    return G


def normpdf3(Ivec, centers, covariances):
    K, d = centers.shape
    G = np.zeros((Ivec.shape[0], K))
    g_k = np.zeros((K, Ivec.shape[0], ))

    for k in range(K):
        diff = Ivec - centers[k]
        delta = np.sum(np.dot(diff, np.linalg.inv(covariances[k])) * diff, -1)
        g_k[k] = (1. / np.sqrt((2 * np.pi) ** 3 *
                  np.linalg.det(covariances[k]))) * np.exp(-0.5 * delta)
    return g_k.T


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
