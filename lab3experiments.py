# %%
import matplotlib.patches as patches
from graphcut_example import graphcut_segm
from norm_cuts_example import *
from mean_shift_example import mean_shift_example
from kmeans_example import kmeans_example
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from PIL import Image
from lab3 import kmeans_segm, mixture_prob

# %%
# Load the image
filename_orange = "images-jpg/orange.jpg"
img_orange = Image.open(filename_orange)
I_orange = np.asarray(img_orange).astype(np.float32)
# Display the image
plt.figure()
plt.imshow(img_orange)
plt.title('Original image')
plt.axis('off')

# %%
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

# %%
# Load the image
filename_tiger1 = "images-jpg/tiger1.jpg"
img_tiger1 = Image.open(filename_tiger1)
I_tiger1 = np.asarray(img_tiger1).astype(np.float32)
# Display the image
plt.figure()
plt.imshow(img_tiger1)
plt.title('Original image')
plt.axis('off')

# %%
K, L = 3, 50
segmentation, centers = kmeans_segm(I_tiger1, K, L)
# Display the segmentation
seg_I = centers[segmentation].reshape(I_tiger1.shape)
seg_img = Image.fromarray(seg_I.astype(np.uint8))
plt.figure()
plt.imshow(seg_img)
plt.title('Segmentation')
plt.axis('off')

# %%
# Load the image
filename_tiger2 = "images-jpg/tiger2.jpg"
img_tiger2 = Image.open(filename_tiger2)
I_tiger2 = np.asarray(img_tiger2).astype(np.float32)
# Display the image
plt.figure()
plt.imshow(img_tiger2)
plt.title('Original image')
plt.axis('off')

# %%
K, L = 4, 50
segmentation, centers = kmeans_segm(I_tiger2, K, L)
# Display the segmentation
seg_I = centers[segmentation].reshape(I_tiger2.shape)
seg_img = Image.fromarray(seg_I.astype(np.uint8))
plt.figure()
plt.imshow(seg_img)
plt.title('Segmentation')
plt.axis('off')

# %%
# Load the image
filename_tiger3 = "images-jpg/tiger3.jpg"
img_tiger3 = Image.open(filename_tiger3)
I_tiger3 = np.asarray(img_tiger3).astype(np.float32)
# Display the image
plt.figure()
plt.imshow(img_tiger3)
plt.title('Original image')
plt.axis('off')

# %%
K, L = 7, 50
segmentation, centers = kmeans_segm(I_tiger3, K, L)
# Display the segmentation
seg_I = centers[segmentation].reshape(I_tiger3.shape)
seg_img = Image.fromarray(seg_I.astype(np.uint8))
plt.figure()
plt.imshow(seg_img)
plt.title('Segmentation')
plt.axis('off')

# %%
K, L = 7, 50
segmentation, centers = kmeans_segm(I_orange, K, L)
seg_I = centers[segmentation].reshape(I_orange.shape)
seg_img = Image.fromarray(seg_I.astype(np.uint8))
plt.imshow(seg_img)


# %%
img = Image.open('Images-jpg/orange.jpg')

K, L = 5, 50
scale_factor = 0.5
sigma = 0.7
kmeans_example(img_orange, K-1, L, scale_factor, sigma)
kmeans_example(img_orange, K, L, scale_factor, sigma)

# %%
K, L = 5, 50
scale_factor = 0.5
sigma = 0.7
kmeans_example(img_tiger1, 3, L, scale_factor, sigma)
kmeans_example(img_tiger1, 5, L, scale_factor, sigma)
kmeans_example(img_tiger1, 10, L, scale_factor, sigma)
kmeans_example(img_tiger1, 20, L, scale_factor, sigma)


# %%
K, L = 5, 50
scale_factor = 0.5
sigma = 2
kmeans_example(img_tiger1, 3, L, scale_factor, sigma)
kmeans_example(img_tiger1, 5, L, scale_factor, sigma)
kmeans_example(img_tiger1, 10, L, scale_factor, sigma)
kmeans_example(img_tiger1, 20, L, scale_factor, sigma)


# %%
K, L = 5, 50
scale_factor = 0.5
sigma = 1
kmeans_example(img_tiger2, 3, L, scale_factor, sigma)
kmeans_example(img_tiger2, 5, L, scale_factor, sigma)
kmeans_example(img_tiger2, 10, L, scale_factor, sigma)


# %%
img_tiger = Image.open('Images-jpg/tiger1.jpg')
img_orange = Image.open('Images-jpg/orange.jpg')

K, L = 5, 10
scale_factor = 0.5
sigma = 0.7

kmeans_example(img_orange, K, L, scale_factor, sigma)

# %% [markdown]
# ## Exercise 3

# %%

img = Image.open('Images-jpg/tiger1.jpg')
# img = Image.open('Images-jpg/tiger2.jpg')

scale = 0.25         # image downscale factor
sigma = 1.0         # image preblurring scale
space_bw = [3, 7, 15]  # , 50]     # spatial bandwidth
colour_bw = [5, 20, 50]    # colour bandwidth
L = 30              # number of mean-shift iterations


f = plt.figure()
f.set_size_inches(40, 20)
#f.subplots_adjust(wspace=-0.5, hspace=0.2)
n_rows = len(space_bw)
n_cols = len(colour_bw)
for i, sigma_s in enumerate(space_bw):
    for j, sigma_c in enumerate(colour_bw):
        f.add_subplot(n_rows, n_cols, i*n_cols+j+1, title="s=" +
                      str(sigma_s)+", c=" + str(sigma_c))

        output = mean_shift_example(img, scale, sigma, sigma_s, sigma_c, L)
        plt.imshow(output)


# %%

img = Image.open('Images-jpg/tiger2.jpg')

scale = 0.25         # image downscale factor
sigma = 1.0         # image preblurring scale
space_bw = [3, 7, 15]     # spatial bandwidth
colour_bw = [5, 20, 50]    # colour bandwidth
L = 30              # number of mean-shift iterations


f = plt.figure()
f.set_size_inches(40, 20)
#f.subplots_adjust(wspace=-0.5, hspace=0.2)
n_rows = len(space_bw)
n_cols = len(colour_bw)
for i, sigma_s in enumerate(space_bw):
    for j, sigma_c in enumerate(colour_bw):
        f.add_subplot(n_rows, n_cols, i*n_cols+j+1, title="s=" +
                      str(sigma_s)+", c=" + str(sigma_c))

        output = mean_shift_example(img, scale, sigma, sigma_s, sigma_c, L)
        plt.imshow(output)


# %%
img = Image.open('Images-jpg/tiger1.jpg')

scale = 0.25         # image downscale factor
# 1.0         # image preblurring scale
sigma = [0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
space_bw = 3     # spatial bandwidth
colour_bw = 20    # colour bandwidth
L = 30              # number of mean-shift iterations


f = plt.figure()
f.set_size_inches(40, 20)
#f.subplots_adjust(wspace=-0.5, hspace=0.2)
for i, blur in enumerate(sigma):
    f.add_subplot(len(sigma)//3, 3, i+1, title="sigma="+str(blur))

    output = mean_shift_example(img, scale, blur, space_bw, colour_bw, L)
    plt.imshow(output)

# %% [markdown]
# ## Exercise 4

# %%


def norm_cut(img, image_sigma, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth):
    img = img.resize(
        (int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))

    h = ImageFilter.GaussianBlur(image_sigma)
    I = np.asarray(img.filter(ImageFilter.GaussianBlur(
        image_sigma))).astype(np.float32)

    segm = norm_cuts_segm(I, colour_bandwidth, radius,
                          ncuts_thresh, min_area, max_depth)
    Inew = mean_segments(img, segm)
    if True:
        Inew = overlay_bounds(img, segm)

    return Inew


# %%
img = Image.open('Images-jpg/tiger1.jpg')
radius = 1               # maximum neighbourhood distance
scale_factor = 0.25      # image downscale factor
image_sigma = 0.5        # image preblurring scale

ncuts_thresh = 0.050      # cutting threshold
min_area = 50           # minimum area of segment
max_depth = 15            # maximum splitting depth
colour_bandwidth = 50.0  # color bandwidth

cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                   radius, ncuts_thresh, min_area, max_depth)
plt.imshow(cut_img)

# %%
img = Image.open('Images-jpg/orange.jpg')
radius = 1               # maximum neighbourhood distance
scale_factor = 0.25      # image downscale factor
image_sigma = 1.0        # image preblurring scale

ncuts_thresh = 0.02      # cutting threshold
min_area = 100           # minimum area of segment
max_depth = 10            # maximum splitting depth
colour_bandwidth = 10.0  # color bandwidth

cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                   radius, ncuts_thresh, min_area, max_depth)
plt.imshow(cut_img)

# %%
img = Image.open('Images-jpg/tiger1.jpg')

ncuts_thresh = 0.05                     # cutting threshold
min_area = [10, 50, 250]                # minimum area of segment
max_depth = 15                          # maximum splitting depth
colour_bandwidth = 50.0                 # color bandwidth

plt.rcParams.update({'font.size': 22})
f = plt.figure()
f.set_size_inches(40, 20)
for i, min_area in enumerate(min_area):
    f.add_subplot(1, 3, i+1, title="min_area="+str(min_area))
    cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                       radius, ncuts_thresh, min_area, max_depth)
    plt.imshow(cut_img)

# %%
ncuts_thresh = [0.005, 0.05, 0.5]       # cutting threshold
min_area = 50                           # minimum area of segment
max_depth = 15                          # maximum splitting depth
colour_bandwidth = 50.0                 # color bandwidth

plt.rcParams.update({'font.size': 22})
f = plt.figure()
f.set_size_inches(40, 20)
for i, thresh in enumerate(ncuts_thresh):
    f.add_subplot(1, 3, i+1, title="ncuts_thresh="+str(thresh))
    cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                       radius, thresh, min_area, max_depth)
    plt.imshow(cut_img)

# %%
ncuts_thresh = 0.05                     # cutting threshold
min_area = 50                           # minimum area of segment
max_depth = [2, 5, 15]                  # maximum splitting depth
colour_bandwidth = 50.0                 # color bandwidth

plt.rcParams.update({'font.size': 22})
f = plt.figure()
f.set_size_inches(40, 20)
for i, depth in enumerate(max_depth):
    f.add_subplot(1, 3, i+1, title="max_depth="+str(depth))
    cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                       radius, ncuts_thresh, min_area, depth)
    plt.imshow(cut_img)

# %%
ncuts_thresh = 0.05                     # cutting threshold
min_area = 50                           # minimum area of segment
max_depth = 9                           # maximum splitting depth
colour_bandwidth = [10.0, 50.0, 90.0]   # color bandwidth

plt.rcParams.update({'font.size': 22})
f = plt.figure()
f.set_size_inches(40, 20)
for i, c_bw in enumerate(colour_bandwidth):
    f.add_subplot(1, 3, i+1, title="colour_bandwidth="+str(c_bw))
    cut_img = norm_cut(img, image_sigma, c_bw, radius,
                       ncuts_thresh, min_area, max_depth)
    plt.imshow(cut_img)

# %%
ncuts_thresh = 0.05                     # cutting threshold
min_area = 50                           # minimum area of segment
max_depth = 9                           # maximum splitting depth
colour_bandwidth = 50.0                 # color bandwidth
radius_list = [1, 2, 3]   # color bandwidth

plt.rcParams.update({'font.size': 22})
f = plt.figure()
f.set_size_inches(40, 20)
for i, radius in enumerate(radius_list):
    f.add_subplot(1, 3, i+1, title="radius="+str(radius))
    cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                       radius, ncuts_thresh, min_area, max_depth)
    plt.imshow(cut_img)

# %%
ncuts_thresh = 0.05                     # cutting threshold
min_area = 50                           # minimum area of segment
max_depth = 9                           # maximum splitting depth
colour_bandwidth = 50.0                 # color bandwidth
radius_list = [1, 2, 3]   # color bandwidth

plt.rcParams.update({'font.size': 22})
f = plt.figure()
f.set_size_inches(40, 20)
for i, radius in enumerate(radius_list):
    new_ncut_thresh = ncuts_thresh*radius
    f.add_subplot(1, 3, i+1, title="radius="+str(radius) +
                  ", ncuts_thresh="+str(round(new_ncut_thresh, 5)))
    cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                       radius, new_ncut_thresh, min_area, max_depth)
    plt.imshow(cut_img)

# %%
ncuts_thresh = 0.05                     # cutting threshold
min_area = 50                           # minimum area of segment
max_depth = 9                           # maximum splitting depth
colour_bandwidth = 50.0                 # color bandwidth
radius_list = [4, 5, 6]   # color bandwidth

plt.rcParams.update({'font.size': 22})
f = plt.figure()
f.set_size_inches(40, 20)
for i, radius in enumerate(radius_list):
    new_ncut_thresh = ncuts_thresh*radius
    f.add_subplot(1, 3, i+1, title="radius="+str(radius) +
                  ", ncuts_thresh="+str(round(new_ncut_thresh, 5)))
    cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                       radius, new_ncut_thresh, min_area, max_depth)
    plt.imshow(cut_img)

# %%
ncuts_thresh = 0.01                     # cutting threshold
min_area = 50                           # minimum area of segment
max_depth = 9                           # maximum splitting depth
colour_bandwidth = 50.0                 # color bandwidth
radius_list = [4, 5, 6]   # color bandwidth

plt.rcParams.update({'font.size': 22})
f = plt.figure()
f.set_size_inches(40, 20)
for i, radius in enumerate(radius_list):
    new_ncut_thresh = ncuts_thresh*radius
    f.add_subplot(1, 3, i+1, title="radius="+str(radius) +
                  ", ncuts_thresh="+str(round(new_ncut_thresh, 5)))
    cut_img = norm_cut(img, image_sigma, colour_bandwidth,
                       radius, new_ncut_thresh, min_area, max_depth)
    plt.imshow(cut_img)

# %% [markdown]
# ## Exercise 5

# %%


def graphcut(img, scale_factor, area, K, alpha, sigma):
    img = img.resize(
        (int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))

    area = [int(i*scale_factor) for i in area]
    I = np.asarray(img).astype(np.float32)
    segm, prior = graphcut_segm(I, area, K, alpha, sigma)

    Inew = mean_segments(img, segm)
    if True:
        Inew = overlay_bounds(img, segm)

    img = Image.fromarray(Inew.astype(np.ubyte))
    return img


# %%


def graphcut_experiment(ADJUST_RECT, scale_factor, area, K, alpha, sigma, img):
    fig, ax = plt.subplots()

    if ADJUST_RECT:
        ax.imshow(img)
    else:
        seg_img = graphcut(img, scale_factor, area, K, alpha, sigma)
        ax.imshow(seg_img)
        area = [int(i*scale_factor) for i in area]

    rect = patches.Rectangle((area[0], area[1]), area[2]-area[0],
                             area[3]-area[1], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    plt.axis('off')
    plt.show()


# %%
ADJUST_RECT = False

scale_factor = 0.5           # image downscale factor
# image region to train foreground with [ minx, miny, maxx, maxy ]
area = [190, 85, 370, 200]
K = 16                       # number of mixture components
alpha = 30.0                  # maximum edge cost
sigma = 10.0                 # edge cost decay factor


graphcut_experiment(ADJUST_RECT, scale_factor, area, K, alpha, sigma, img)

# %%
ADJUST_RECT = False

scale_factor = 0.5           # image downscale factor
# image region to train foreground with [ minx, miny, maxx, maxy ]
area = [190, 85, 410, 300]
K = 1                       # number of mixture components
alpha = 200.0                  # maximum edge cost
sigma = 200.0                 # edge cost decay factor

# img = Image.open('Images-jpg/tiger1.jpg') # area = [ 80, 150, 570, 300 ], alpha = 8, sigma = 30
img = Image.open('Images-jpg/tiger2.jpg')

graphcut_experiment(ADJUST_RECT, scale_factor, area, K, alpha, sigma, img)

# %%
ADJUST_RECT = False
scale_factor = 0.5           # image downscale factor
# image region to train foreground with [ minx, miny, maxx, maxy ]
area = [100, 170, 490, 280]
K = 16                       # number of mixture components
alpha = 20.0                  # maximum edge cost
sigma = 20.0                 # edge cost decay factor

# img = Image.open('Images-jpg/tiger1.jpg') # area = [ 80, 150, 570, 300 ], alpha = 8, sigma = 30
img = Image.open('Images-jpg/tiger1.jpg')
graphcut_experiment(ADJUST_RECT, scale_factor, area, K, alpha, sigma, img)


# %%
