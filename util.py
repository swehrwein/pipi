import numpy as np
import scipy as sp
import skimage as skim
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip

import sys
import timeit
import warnings


def rgb2gray(a):
    #print a
    if a.ndim == 3:
        return skim.color.rgb2gray(a)
    elif a.ndim == 4:
        result = np.zeros((a.shape[:2] + (a.shape[3],)))
        for i in range(a.shape[-1]):
            result[...,i] = skim.color.rgb2gray(a[...,i])
        return result
    elif a.ndim == 1 and a.size == 3:
        return skim.color.rgb2gray(a[np.newaxis, np.newaxis, :])
    elif a.ndim == 2:
        return a
    else:
        print "rgb2gray: bad number of dimensions"


def gray2rgb(a):
    result = skimage.color.gray2rgb(a)
    if result.ndim == 4:
        return result.transpose((0,1,3,2))
    elif result.ndim == 3:
        return result
    else:
        print "gray2rgb: a must have 2 or 3 dimensions"


def hsv2rgb(a):
    if a.ndim == 3:
        return skimage.color.hsv2rgb(a)
    elif a.ndim == 4:
        for i in range(a.shape[-1]):
            a[...,i] = skimage.color.hsv2rgb(a[...,i])
        return a
    else:
        print "hsv2rgb: array should have 3 or 4 dimensions"


def imresize(img, scale, interp=None):
    if img.dtype == np.uint8:
        if interp is None:
            interp = 'bilinear'
        return sp.misc.imresize(img, scale, interp=interp)
    else:
        scales = [scale, scale]
        if img.ndim == 3:
            scales.append(1.0)

        return sp.ndimage.interpolation.zoom(img, tuple(scales))


def imread(fn, gamma=1.0):
    if fn[-3:] == 'exr':
        im = skim.io.imread(fn, plugin='openexr')
        return np.power(im / np.max(im), gamma)
    else:
        return np.power(skim.img_as_float(skim.io.imread(fn)), gamma)


def imwrite(im, fn, dtype=np.float16):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if im.dtype == bool:
            im = im.astype('float')
        if fn[-3:] == 'exr':
            skim.io.imsave(fn, im.astype(dtype), plugin='openexr')
        else:
            skim.io.imsave(fn, im)


def imshow(im, colorbar=False, **kwargs):
    if im.ndim == 3 and im.shape[2] == 2:
        im = np.dstack((im[:,:,0], im[:,:,1], np.zeros(im.shape[:2])))
    skim.io.imshow(im, **kwargs)
    plt.colorbar()
    skim.io.show()


def implay(array):
    try:
        if array.dtype != np.uint8:
            array = (array*255).astype(np.uint8)

        def make_frame(t):
            return array[...,round(t*30.0)]

        clip = VideoClip(make_frame, duration=array.shape[-1] / 30.0)
        clip.preview(fps=30)

        #_ = raw_input("quit...")
    finally:
        import pygame.display
        pygame.display.quit()



def hist(data, *args, **kwargs):
    if data.ndim > 1:
        data = np.reshape(data, (-1,))
    plt.hist(data, *args)
    plt.show()


def div_nonz(a,b):
    anz = a[b != 0]
    bnz = b[b != 0]
    result = np.zeros_like(a)
    result[b != 0] = anz / bnz
    return result


def norm(arr, axis=-1):
    return np.sqrt(np.sum(arr**2, axis=axis))


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


def mldivide(A,B):
    """
    Solves Ax = B for x; equivalent to matlab's A\B
    Conceptually similar to X = inv(A)*B
    """
    return np.linalg.lstsq(A,B)


def mrdivide(A,B):
    """
    Solves xA = B for x; equivalent to matlab's B/A
    Conceptually similar to X = B*inv(A)
    """
    X, residuals, rank, s = np.linalg.lstsq(A.T, B.T)
    return (X.T, residuals, rank, s)


def vec(a):
    return np.reshape(a, (-1,))


def unstuff(a, mask):
    if a.ndim == 3:
        return a[mask,:]
    elif a.ndim == 2:
        return a[mask]


def stuff(a, mask):
    if a.ndim == 1:
        result = np.zeros(mask.shape)
        result[mask] = a
    elif a.ndim == 2:
        result = np.zeros(mask.shape + (a.shape[1],))
        result[mask,:] = a
    return result


def azel2enu(az, el):
    e = np.cos(el) * np.sin(az)
    n = np.cos(el) * np.cos(az)
    u = np.sin(el)
    return np.array([e,n,u])


def enu2azel(e,n,u):
    az = (np.arctan2(e,n) + 2*np.pi) % (2*np.pi)
    el = np.arctan2(u, np.sqrt(e**2 + n**2))
    return az, el


class Opt(object):
    def __init__(self, **args):
        self.update(**args)

    def __str__(self):
        return '\n'.join("%s: %s" % (k,v) for (k,v) in vars(self).iteritems())

    def update(self, **args):
        self.__dict__.update(args)

    def asdict(self):
        return vars(self)



class Timer(object):
    def __init__(self, msg='', printout=True):
        self.msg = msg
        self.printout = printout

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
