import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t, return_gaussians = False):
    [i, j] = np.shape(pic)
    Fhat = fft2(pic)


    x = np.arange(-i/2 + 1, i/2 + 1)
    y = np.arange(-j/2 + 1, j/2 + 1)
    X, Y = np.meshgrid(x, y)

    # Factor t in the scaling factor?
    gauss = 1/(2*np.pi*t) * np.exp(-(np.square(X) + np.square(Y))/(2*t))

    # [x, y] = np.meshgrid(np.linspace(0, 1-1/w, w),np.linspace(0, 1-1/h, h))
    # gauss = 1/(2*np.pi*t) * np.exp(-(np.square(x) + np.square(y))/(2*t))

    Ghat = fft2(gauss)

    
    # Ghat = fftshift(fft2(gauss))
    
    result = fftshift(ifft2(Fhat*Ghat))

    # For debug and testing
    if return_gaussians:
        return result, gauss, np.real(Ghat)

    return result


