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
if 0:
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


# Exercise 2.3

if 0:
	show_pulse_blurs = False
	show_image_blurs = True
	
	t = 10 	# 0.1, 0.3, 1.0, 10.0, 100.0

	# img = np.load("Images-npy/phonecalc128.npy")
	img = np.load("Images-npy/kaffe256.npy")
	
	blurred_img = gaussfft(img, t)

	var = variance(blurred_img)
	print(var)

	f = plt.figure()
	f.subplots_adjust(wspace=0.1, hspace=0.2)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(1, 2, 1)
	showgrey(img, False)
	a1.title.set_text("Original image")

	a2 = f.add_subplot(1, 2, 2)
	showgrey(blurred_img, False)
	a2.title.set_text("Image blurred with t = {}".format(t))
	
	plt.show()

	# CODE FOR SHOWING IMAGE BLURS
	if show_image_blurs:
		f = plt.figure()
		f.subplots_adjust(wspace=0.1, hspace=0.2)
		plt.rc('axes', titlesize=10)

		a1 = f.add_subplot(3, 2, 1)
		showgrey(img, False)
		a1.title.set_text("Original image")

		a2 = f.add_subplot(3, 2, 2)
		showgrey(gaussfft(img, 1.0), False)
		a2.title.set_text("Image blurred with t = 1.0")
		
		a3 = f.add_subplot(3, 2, 3)
		showgrey(gaussfft(img, 4.0), False)
		a3.title.set_text("Image blurred with t = 4.0")
		
		a4 = f.add_subplot(3, 2, 4)
		showgrey(gaussfft(img, 16.0), False)
		a4.title.set_text("Image blurred with t = 16.0")
		
		a5 = f.add_subplot(3, 2, 5)
		showgrey(gaussfft(img, 64.0), False)
		a5.title.set_text("Image blurred with t = 64.0")
		
		a6 = f.add_subplot(3, 2, 6)
		showgrey(gaussfft(img, 256.0), False)
		a6.title.set_text("Image blurred with t = 256.0")
		
		plt.show()

	# CODE FOR SHOWING ALL IMPULSE BLURS
	if show_pulse_blurs:
		f = plt.figure()
		f.subplots_adjust(wspace=0.1, hspace=0.2)
		plt.rc('axes', titlesize=10)

		a1 = f.add_subplot(3, 2, 1)
		showgrey(deltafcn(128, 128), False)
		a1.title.set_text("Original pulse")

		a2 = f.add_subplot(3, 2, 2)
		showgrey(gaussfft(deltafcn(128, 128), 0.1), False)
		a2.title.set_text("Image blurred with t = 0.1")
		
		a3 = f.add_subplot(3, 2, 3)
		showgrey(gaussfft(deltafcn(128, 128), 0.3), False)
		a3.title.set_text("Image blurred with t = 0.3")
		
		a4 = f.add_subplot(3, 2, 4)
		showgrey(gaussfft(deltafcn(128, 128), 1.0), False)
		a4.title.set_text("Image blurred with t = 1.0")
		
		a5 = f.add_subplot(3, 2, 5)
		showgrey(gaussfft(deltafcn(128, 128), 10.0), False)
		a5.title.set_text("Image blurred with t = 10.0")
		
		a6 = f.add_subplot(3, 2, 6)
		showgrey(gaussfft(deltafcn(128, 128), 100.0), False)
		a6.title.set_text("Image blurred with t = 100.0")
		
		plt.show()


# Exercise 3.1
if 0:
	office = np.load("Images-npy/office256.npy")
	add = gaussnoise(office, 16)
	sap = sapnoise(office, 0.1, 255)

	# showgrey(office)
	# showgrey(add)
	# showgrey(sap)

	# Gaussian noise 
	add_smooth = gaussfft(add, 9)
	add_median = medfilt(add, 5)
	add_lowpass = ideal(add, 0.2)

	# Salt and pepper noise
	sap_smooth = gaussfft(sap, 9)
	sap_median = medfilt(sap, 5)
	sap_lowpass = ideal(sap, 0.2)

	f = plt.figure()
	f.subplots_adjust(wspace=0.1, hspace=0.2)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(1, 3, 1)
	showgrey(office, False)
	a1.title.set_text("Original image")

	a2 = f.add_subplot(1, 3, 2)
	showgrey(add, False)
	a2.title.set_text("Image with gaussian noise")

	a3 = f.add_subplot(1, 3, 3)
	showgrey(add_lowpass, False)
	a3.title.set_text("Smoothed with lowpass (cutoff frequency 0.2)")


	# a1 = f.add_subplot(3, 2, 2)
	# showgrey(add_smooth, False)
	# a1.title.set_text("Gauss smoothed")

	# a1 = f.add_subplot(3, 2, 3)
	# showgrey(add_median, False)
	# a1.title.set_text("Median smoothed")

	# a1 = f.add_subplot(3, 2, 4)
	# showgrey(add_lowpass, False)
	# a1.title.set_text("Low-pass smoothed")

	plt.show()



# Exercise 3.2

if 1:
	img = np.load("Images-npy/phonecalc256.npy")
	smoothimg = img
	N = 5
	f = plt.figure()
	f.subplots_adjust(wspace=0, hspace=0)
	for i in range(N):
		if i>0: # generate subsampled versions
			img = rawsubsample(img)
			smoothimg = ideal(smoothimg, 0.3)
			smoothimg = rawsubsample(smoothimg)
		f.add_subplot(2, N, i + 1)
		showgrey(img, False)
		f.add_subplot(2, N, i + N + 1)
		showgrey(smoothimg, False)
	plt.show()