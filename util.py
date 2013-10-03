import numpy as np
import skimage as skim
import skimage.io
import skimage.color

import sys
import timeit
import warnings

def rgb2gray(a):
    #print a
    if a.ndim == 3:
        return skim.color.rgb2gray(a)
    elif a.ndim == 1 and a.size ==3:
        return skim.color.rgb2gray(a[np.newaxis, np.newaxis, :])
    elif a.ndim == 2:
        return a

def imread(f, gamma=1.0):
    if f[-3:] == 'exr':
        im = skim.io.imread(f, plugin='openexr')
        return np.power(im / np.max(im), gamma)
    else:
        return np.power(skim.img_as_float(skim.io.imread(f)), gamma)

def imwrite(im, fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if im.dtype == bool:
            im = im.astype('float')
        skim.io.imsave(fn, im)

def imshow(im, **kwargs):
    skim.io.imshow(im, **kwargs)
    skim.io.show()

def div_nonz(a,b):
    anz = a[b != 0]
    bnz = b[b != 0]
    result = np.zeros_like(a)
    result[b != 0] = anz / bnz
    return result

def bilerp(im, r, c):
    r = np.asarray(r)
    c = np.asarray(c)

    r0 = np.floor(r).astype(int)
    r1 = r0 + 1
    c0 = np.floor(c).astype(int)
    c1 = c0 + 1

    Ia = im[r0, c0]
    Ib = im[r0, c1]
    Ic = im[r1, c0]
    Id = im[r1, c1]

    wa = (r1-r) * (c1-c)
    wb = (r1-r) * (c-c0)
    wc = (r-r0) * (c1-c)
    wd = (r-r0) * (c-c0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def mmr(arr):
    return np.min(arr), np.max(arr), np.ptp(arr)
def pmmr(arr):
    print '%f %f %f' % mmr(arr)

class Opt(object):
    pass

class Timer(object):
    def __init__(self, msg='', printout=True):
        self.msg = msg
        self.printout=printout

    def __enter__(self):
        if self.printout:
            print self.msg,
            sys.stdout.flush()
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        if self.printout:
            print '%.2f' % self.elapsed
            sys.stdout.flush()

