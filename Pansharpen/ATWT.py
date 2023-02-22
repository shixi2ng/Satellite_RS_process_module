import numpy as np
import math
from numpy.ctypeslib import ndpointer
import ctypes
from scipy.special import betainc
from numpy.fft import fft2, ifft2, fftshift
import scipy.ndimage.interpolation as ndii


class ATWTArray(object):
    '''A trous wavelet transform'''
    def __init__(self,band):
        self.num_iter = 0
#      cubic spline filter
        self.H = np.array([1.0/16,1.0/4,3.0/8,1.0/4,1.0/16])
#      data arrays
        self.lines,self.samples = band.shape
        self.bands = np.zeros((4,self.lines,self.samples),np.float32)
        self.bands[0,:,:] = np.asarray(band,np.float32)

    def inject(self,band):
        m = self.lines
        n = self.samples
        self.bands[0,:,:] = band[0:m,0:n]

    def get_band(self,i):
        return self.bands[i,:,:]

    def normalize(self,a,b):
        if self.num_iter > 0:
            for i in range(1,self.num_iter+1):
                self.bands[i,:,:] = a*self.bands[i,:,:]+b

    def filter(self):
        if self.num_iter < 3:
            self.num_iter += 1
#          a trous filter
            n = 2**(self.num_iter-1)
            H = np.vstack((self.H,np.zeros((2**(n-1),5))))
            H = np.transpose(H).ravel()
            H = H[0:-n]
#          temporary arrays
            f1 = np.zeros((self.lines,self.samples))
            ff1 = f1*0.0
#          filter columns
            f0 = self.bands[0,:,:]
#          filter columns
            for i in range(self.samples):
                f1[:,i] = np.convolve(f0[:,i].ravel(), H, 'same')
#          filter rows
            for j in range(self.lines):
                ff1[j,:] = np.convolve(f1[j,:], H, 'same')
            self.bands[self.num_iter,:,:] = self.bands[0,:,:] - ff1
            self.bands[0,:,:] = ff1

    def invert(self):
        if self.num_iter > 0:
            self.bands[0,:,:] += self.bands[self.num_iter,:,:]
            self.num_iter -= 1


def orthoregress(x,y):
    Xm = np.mean(x)
    Ym = np.mean(y)
    s = np.cov(x,y)
    R = s[0,1]/math.sqrt(s[1,1]*s[0,0])
    lam,vs = np.linalg.eig(s)
    idx = np.argsort(lam)
    vs = vs[:,idx]      # increasing order, so
    b = vs[1,1]/vs[0,1] # first pc is second column
    return [b,Ym-b*Xm,R]