# %% [markdown]
# # Lab 2
# DD2423 Image Analysis and Computer Vision

# %% [markdown]
# ## Exercise 1

# %%
import numpy as np
#from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft

# %%


def deltax():
    # This version ensures both dxtools and dytools have the same size
    dxmask = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    return dxmask


def sobelx():
    sobelxmask = 1/4*np.array([[1], [2], [1]])*1/2*np.array([[1, 0, -1]])
    return sobelxmask


def deltay():
    dymask = np.array([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]])
    return dymask


def sobely():
    sobelymask = 1/4*np.array([[1, 2, 1]])*1/2*np.array([[1], [0], [-1]])
    return sobelymask


# %%
tools = np.load("Images-npy/few256.npy")
dxtools = convolve2d(tools, deltax(), 'valid')
dytools = convolve2d(tools, deltay(), 'valid')
sobelxtools = convolve2d(tools, sobelx(), 'valid')
sobelytools = convolve2d(tools, sobely(), 'valid')

showgrey(tools)
showgrey(dxtools)
showgrey(dytools)
showgrey(sobelxtools)
showgrey(sobelytools)
showgrey(dxtools-sobelxtools)

print("(h, w):", tools.shape, dxtools.shape, dytools.shape)

# %% [markdown]
# ## Exercise 2

# %%
# Lv returns the gradient magnitude in every point of the image.
# If a nonzero parameter t is provided, the image is filtered with gaussian of variance t


def Lv(inpic, t=0, shape='same'):
    if t:
        inpic = gaussfft(inpic, t)
    Lx = convolve2d(inpic, deltax(), shape)
    Ly = convolve2d(inpic, deltay(), shape)
    return np.sqrt(Lx**2 + Ly**2)


threshold = 20
blur_threshold = 15

tools = np.load("Images-npy/few256.npy")
showgrey(tools)
showgrey((Lv(tools) > threshold).astype(int))
showgrey((Lv(tools, t=1.0) > blur_threshold).astype(int))

godthem = np.load("Images-npy/godthem256.npy")
showgrey(godthem)
showgrey((Lv(godthem) > threshold).astype(int))
# Gaussian blur applied prior to gradient calculation
showgrey((Lv(godthem, t=1.0) > blur_threshold).astype(int))


# %% [markdown]
# ## Exercise 4

# %%
def dxmask():
    dx = np.zeros([5, 5])
    dx[2, 1] = 0.5
    dx[2, 3] = -0.5
    return dx


def dymask():
    dy = np.zeros([5, 5])
    dy[1, 2] = 0.5
    dy[3, 2] = -0.5
    return dy


def dxxmask():
    dxx = np.zeros([5, 5])
    dxx[2, 1] = 1
    dxx[2, 2] = -2
    dxx[2, 3] = 1
    return dxx


def dyymask():
    dyy = np.zeros([5, 5])
    dyy[1, 2] = 1
    dyy[2, 2] = -2
    dyy[3, 2] = 1
    return dyy


def dxymask():
    return convolve2d(dxmask(), dymask(), 'same')


def dxxxmask():
    return convolve2d(dxmask(), dxxmask(), 'same')


def dxxymask():
    return convolve2d(dxxmask(), dymask(), 'same')


def dxxymask2():
    return convolve2d(dxmask(), dxymask(), 'same')


def dxyymask():
    return convolve2d(dxmask(), dyymask(), 'same')


def dyyymask():
    return convolve2d(dymask(), dyymask(), 'same')


def Lvvtilde(inpic, shape="same"):
    Lx = convolve2d(inpic, dxmask(), shape)
    Lxx = convolve2d(inpic, dxxmask(), shape)
    Ly = convolve2d(inpic, dymask(), shape)
    Lxy = convolve2d(inpic, dxymask(), shape)
    Lyy = convolve2d(inpic, dyymask(), shape)

    return Lx**2 * Lxx + 2 * Lx * Ly * Lxy + Ly**2 * Lyy


def Lvvvtilde(inpic, shape="same"):
    Lx = convolve2d(inpic, dxmask(), shape)
    Ly = convolve2d(inpic, dymask(), shape)
    Lxxx = convolve2d(inpic, dxxxmask(), shape)
    Lxxy = convolve2d(inpic, dxxymask(), shape)
    Lxyy = convolve2d(inpic, dxyymask(), shape)
    Lyyy = convolve2d(inpic, dyyymask(), shape)

    return Lx**3 * Lxxx + 3 * Lx**2 * Ly * Lxxy + 3 * Lx * Ly**2 * Lxyy + Ly**3 * Lyyy


[x, y] = np.meshgrid(range(-5, 6), range(-5, 6))
print(convolve2d(x**3, dxxxmask(), 'valid'))      # dxxx(x^3) = 6
print(convolve2d(x**3, dxxmask(), 'valid'))       # dxx(x^3) = 6x
print(convolve2d(x**2*y, dxxymask(), 'valid'))    # dxxy(x^2 * y) = 2
print(convolve2d(x*y**2, dxxmask(), 'valid'))     # dxx(x*y^2) = 0

# %%
scale = 16.0
house = np.load("Images-npy/godthem256.npy")
showgrey(house)
showgrey(contour(Lvvtilde(discgaussfft(house, scale), 'same')))

# %%
scale = 16.0
tools = np.load("Images-npy/few256.npy")
showgrey(tools)
showgrey((Lvvvtilde(discgaussfft(tools, scale), 'same') < 0).astype(int))

# %%


def diff_geom_edge_detection(inpic, scale):
    return contour(Lvvtilde(discgaussfft(inpic, scale), 'same')) * (Lvvvtilde(discgaussfft(inpic, scale), 'same') < 0).astype(int)


inpic = tools
showgrey(inpic)
showgrey(diff_geom_edge_detection(inpic, 4.0))

# %%

scales = [1, 4, 16, 64]
n_cols = len(scales)
n_rows = 5
f = plt.figure()

inpic = house

f.set_size_inches(10, 15)
f.subplots_adjust(wspace=0.1, hspace=-0.4)
for i, scale in enumerate(scales):
    f.add_subplot(n_rows, n_cols, i+1, title="scale = " + str(scale))
    showgrey(discgaussfft(inpic, scale), display=False)

    f.add_subplot(n_rows, n_cols, i+1+n_cols, title="Lvvtilde=0")
    showgrey(contour(Lvvtilde(discgaussfft(inpic, scale), 'same')), display=False)

    f.add_subplot(n_rows, n_cols, i+1+2*n_cols, title="Lvvvtilde<0")
    showgrey((Lvvvtilde(discgaussfft(inpic, scale), 'same')
             < 0).astype(int), display=False)

    f.add_subplot(n_rows, n_cols, i+1+3*n_cols, title="edge detection")
    showgrey(diff_geom_edge_detection(inpic, scale), display=False)

    f.add_subplot(n_rows, n_cols, i+1+4*n_cols, title="difference")
    showgrey(diff_geom_edge_detection(inpic, scale) -
             contour(Lvvtilde(discgaussfft(inpic, scale), 'same')), display=False)


# %% [markdown]
# # Exercise 5

# %%
def extractedge(inpic, scale, threshold=0, shape='same'):
    if scale > 0:
        smoothed = gaussfft(inpic, scale)
    else:
        smoothed = inpic
    Lvv = Lvvtilde(smoothed, shape)
    Lvvv = Lvvvtilde(smoothed, shape)

    # Returns zero-crossings of Lvv for which Lvvv < 0 is true
    curves = zerocrosscurves(Lvv, Lvvv < 0)

    if threshold:
        # Removes the points for which second argument is not true (gradient magnitude below threshold)
        curves = thresholdcurves(curves, Lv(smoothed) > threshold)

    return curves


# %%
image = np.load("Images-npy/few256.npy")
# image = np.load("Images-npy/godthem256.npy")

scale = 4
threshold = 6


edgecurves = extractedge(image, scale, threshold, 'same')

overlaycurves(image, edgecurves)


# %%
image = np.load("Images-npy/godthem256.npy")

scale = 6
threshold = 3


edgecurves = extractedge(image, scale, threshold, 'same')

overlaycurves(image, edgecurves)


# %%
scales = [0.0001, 1, 4, 16, 64]
thresholds = [0, 2, 5, 10]
img = house
f = plt.figure()
f.set_size_inches(15, 15)
f.subplots_adjust(wspace=-0.5, hspace=0.2)
n_rows = len(scales)
n_cols = len(thresholds)
for i, scale in enumerate(scales):
    for j, threshold in enumerate(thresholds):
        f.add_subplot(n_rows, n_cols, i*n_cols+j+1, title="s=" +
                      str(scale)+", t=" + str(threshold))
        edgecurves = extractedge(img, scale, threshold, 'same')
        overlaycurves(img, edgecurves)
        # plt.axis('off')

# %% [markdown]
# ## Exercise 6

# %%


def houghline(curves, smoothed, nrho, ntheta, threshold, nlines, verbose, blur_acc=0, acc_func=lambda x: 1, d_deriv=False):

    shape = "same"
    magnitude = Lv(smoothed, shape=shape)
    Lx = convolve2d(inpic, deltax(), shape)
    Ly = convolve2d(inpic, deltay(), shape)

    # Allocate accumulator space
    acc = np.zeros((nrho, ntheta))
    linepar = []

    # center curves
    Y, X = curves
    ALL_IMAGE = True
    if ALL_IMAGE:
        n, m = smoothed.shape  # we don't need to look that far
        dY, dX = - n//2, - m//2
        Y_C, X_C = Y + dY, X + dX
        max_dist = np.sqrt((n/2)**2 + (m/2)**2)
    else:
        # only as far as there are edge points
        n, m = Y.max()-Y.min()+1, X.max()-X.min()+1
        # center
        dY, dX = - Y.min() - n//2, - X.min() - m//2
        Y_C, X_C = Y + dY, X + dX
        max_dist = np.sqrt(X_C**2 + Y_C**2).max()

    # Define a coordinate system in the accumulator space
    rho_space = np.linspace(0, max_dist, nrho)
    rho_step = (rho_space[-1] - rho_space[0])/(nrho-1)
    theta_space = np.linspace(0, 2*np.pi, ntheta)

    # Loop over all the edge points
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x_c, y_c = X_C[i], Y_C[i]

        # Check if valid point with respect to threshold
        if magnitude[y, x] < threshold:
            continue

        # Optionally, keep value from magnitude image
        # Loop over a set of theta values
        for theta_idx in range(ntheta):
            theta = theta_space[theta_idx]
            # Compute rho for each theta value
            rho = x_c * np.cos(theta) + y_c * np.sin(theta)
            rho_idx = int(np.round((rho - rho_space[0])/rho_step))
            if rho_idx < 0 or rho_idx >= nrho:  # point will be detected with opposite theta
                continue
                # rho = -rho
                # theta = theta + np.pi
            # Compute index values in the accumulator space
            # rho_idx = np.argmin(np.abs(rho_space - rho))
            # Update the accumulator
            EPS = 1e-6
            grad_mag = magnitude[y, x]
            dderiv_edge_normal = np.abs(
                Lx[y, x] * np.cos(theta) + Ly[y, x] * np.sin(theta)) + EPS  # supposed to be high
            # supposed to be low
            dderiv_edge = np.abs(
                Lx[y, x] * np.sin(theta) - Ly[y, x] * np.cos(theta)) + EPS
            # acc[rho_idx, theta_idx] += grad_edge_normal/grad_edge * acc_func(grad_mag)
            val = grad_mag if not d_deriv else dderiv_edge_normal / \
                (1+dderiv_edge)
            acc[rho_idx, theta_idx] += acc_func(val)

    # Optionally blur the accumulator
    if blur_acc != 0:
        acc = discgaussfft(acc, blur_acc)

    # Extract local maxima from the accumulator
    pos, value, _ = locmax8(acc)
    # Delimit the number of responses if necessary
    indexvector = np.argsort(value)[-nlines:]
    pos = pos[indexvector]
    # Compute a line for each one of the strongest responses in the accumulator
    for idx in range(nlines):
        thetaidxacc = pos[idx, 0]
        rhoidxacc = pos[idx, 1]
        rho = rho_space[rhoidxacc]
        theta = theta_space[thetaidxacc]
        linepar.append([rho, theta])

    if verbose >= 1:
        # Overlay these curves on the gradient magnitude image
        # showgrey(magnitude, display=False)
        # plt.plot(X, Y, 'b.')
        for rho, theta in linepar:
            x0_c, y0_c = rho * np.cos(theta), rho * np.sin(theta)
            x0, y0 = x0_c - dX, y0_c - dY
            dx, dy = np.sin(theta), -np.cos(theta)  # attention!
            plt.plot([x0 - dx*m, x0, x0 + dx*m],
                     [y0 - dy*n, y0, y0 + dy*n], 'r-')
        plt.show()

    if verbose >= 2:
        # diaply the accumulator
        plt.figure()
        plt.imshow(acc, cmap='gray')
        plt.title("Hough space histogram")
        plt.xlabel("i_theta")
        plt.ylabel("i_rho")
        plt.show()

    # Return the output data [linepar, acc]
    return [linepar, acc]


def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose, blur_acc=0, acc_func=lambda x: 1, d_deriv=False):
    # Generate curves for houghline
    curves = extractedge(pic, scale, gradmagnthreshold, 'same')

    # For comparison with generated lines
    if verbose >= 1:
        overlaycurves(pic, curves)

    # Generate magnitude for houghline
    smoothed = discgaussfft(pic, scale)

    return houghline(curves, smoothed, nrho, ntheta, gradmagnthreshold, nlines, verbose, blur_acc, acc_func, d_deriv)


# %%

testimage1 = np.load("Images-npy/triangle128.npy")
smalltest1 = binsubsample(testimage1)

testimage2 = np.load("Images-npy/houghtest256.npy")
smalltest2 = binsubsample(binsubsample(testimage2))

houghedgeline(smalltest2, 2, 10, 100, 100, 10, 2)
houghedgeline(smalltest1, 2, 10, 100, 100, 10, 2)


# %%
houghedgeline(house, 2, 10, 200, 200, 10, 2, 0)

# %%
houghedgeline(testimage1, 0, 10, 100, 100, 1, 2, 0)
houghedgeline(testimage1, 0, 10, 100, 100, 3, 1, 0)

# %%

houghedgeline(testimage2, 0, 10, 100, 100, 1, 2, 0)
houghedgeline(testimage2, 0, 10, 100, 100, 7, 1, 0)

# %% [markdown]
# ### Exercise 6.2

# %%
# houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose, blur_acc=0, acc_func=lambda x: 1):

houghedgeline(house, 2, 5, 200, 200, 10, 2, 0)

houghedgeline(house, 2, 5, 200, 200, 10, 2, 0, lambda x: np.log(1+x))

houghedgeline(house, 2, 5, 200, 200, 10, 2, 0, lambda x: np.sqrt(x))

houghedgeline(house, 2, 5, 200, 200, 10, 2, 0, lambda x: x)

houghedgeline(house, 2, 5, 200, 200, 10, 2, 0, lambda x: x**2)


# %%
houghedgeline(tools, 2, 5, 200, 200, 10, 2, 0)

houghedgeline(tools, 2, 5, 200, 200, 10, 2, 0, lambda x: np.log(1+x))

houghedgeline(tools, 2, 5, 200, 200, 10, 2, 0, lambda x: np.sqrt(x))

houghedgeline(tools, 2, 5, 200, 200, 10, 2, 0, lambda x: x)

houghedgeline(tools, 2, 5, 200, 200, 10, 2, 0, lambda x: x**2)


# %%
houghedgeline(house, 2, 5, 200, 200, 15, 2, 0, lambda x: 1/x)

houghedgeline(house, 2, 5, 200, 200, 15, 2, 0, lambda x: 1/np.log(x+1))

houghedgeline(house, 2, 5, 200, 200, 15, 2, 0, lambda x: 1/x**2)

houghedgeline(house, 2, 5, 200, 200, 15, 2, 0, lambda x: 1/np.sqrt(x))


# %%
houghedgeline(house, 2, 5, 200, 200, 10, 2, 0, lambda x: x, d_deriv=True)
houghedgeline(house, 2, 5, 200, 200, 10, 2, 0,
              lambda x: np.log(x+1), d_deriv=True)
houghedgeline(house, 2, 5, 200, 200, 10, 2, 0,
              lambda x: np.log(1+np.log(x+1)), d_deriv=True)


# %%
