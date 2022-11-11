import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d

from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave


# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.

# Exercise 1.3
if 0:
	# fftwave(5, 9)
	# fftwave(9, 5)
	# fftwave(17, 9)
	# fftwave(17, 121)
	fftwave(5, 1)
	fftwave(125, 1)
	
# Exercise 1.4
if 0:
	F = np.concatenate([np.zeros((56,128)), np.ones((16,128)), np.zeros((56,128))])
	G = F.T
	H = F + 2*G

	
	Fhat = fft2(F)
	Ghat = fft2(G)
	Hhat = fft2(H)

	showgrey(F)
	showgrey(np.log(1 + np.abs(Fhat)))
	showgrey(G)
	showgrey(np.log(1 + np.abs(Ghat)))
	showgrey(H)
	showgrey(np.log(1 + np.abs(Hhat)))
	showgrey(np.log(1 + np.abs(fftshift(Hhat))))

# Exercise 1.5
if 0:
	F = np.concatenate([np.zeros((56,128)), np.ones((16,128)), np.zeros((56,128))])
	G = F.T
	showgrey(F * G)
	showfs(fft2(F * G))
	#showfs(convolve2d(fft2(F),fft2(G)))

# Exercise 1.6 & 1.7
if 0:
	F = np.concatenate([np.zeros((60,128)), np.ones((8,128)), np.zeros((60,128))]) * np.concatenate([np.zeros((128,48)), np.ones((128,32)), np.zeros((128,48))], axis=1)
	showgrey(F)
	showfs(fft2(F))


# Exercise 1.7
if 0:
	alpha = 60

	F = np.concatenate([np.zeros((60,128)), np.ones((8,128)), np.zeros((60,128))]) * np.concatenate([np.zeros((128,48)), np.ones((128,32)), np.zeros((128,48))], axis=1)
	G = rot(F, alpha) # alpha degree rotation
	Ghat = fft2(G)

	Hhat = rot(fftshift(Ghat), -alpha)

	f = plt.figure()
	f.subplots_adjust(wspace=0.1, hspace=0.2)
	plt.rc('axes', titlesize=10)


	a1 = f.add_subplot(3, 2, 1)
	showgrey(F, False)
	a1.title.set_text("F")

	a2 = f.add_subplot(3, 2, 2)
	showfs(fft2(F), False)
	a2.title.set_text("Fhat")

	a3 = f.add_subplot(3, 2, 3)
	showgrey(G, False)
	a3.title.set_text("G (F rotated by %s degrees" % alpha)

	a4 = f.add_subplot(3, 2, 4)
	showfs(Ghat, False)
	a4.title.set_text("Ghat")

	a5 = f.add_subplot(3, 2, 6)
	showgrey(np.log(1 + abs(Hhat)), False)
	a5.title.set_text("Ghat rotated back -%s degrees" % alpha)

	plt.show()



# Exercise 1.8 
if 1:
	img_1 = np.load("Images-npy/phonecalc128.npy")
	img_2 = np.load("Images-npy/few128.npy")
	img_3 = np.load("Images-npy/nallo128.npy")
	
	# Default second param a = 0.001
	a = 0.001
	pow_1 = pow2image(img_1, a)
	pow_2 = pow2image(img_2, a)
	pow_3 = pow2image(img_3, a)

	plt.figure(1)
	showgrey(pow_1, False)
	plt.figure(2)
	showgrey(randphaseimage(pow_1), False)
	plt.show()

	plt.figure(1)
	showgrey(pow_2, False)
	plt.figure(2)
	showgrey(randphaseimage(pow_2), False)
	plt.show()

	plt.figure(1)
	showgrey(pow_3, False)
	plt.figure(2)
	showgrey(randphaseimage(pow_3), False)
	plt.show()


# Exercise 2
	
	