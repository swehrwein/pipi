import itertools
import numpy as np
import sys
#import magic
import gzip


class BundleFile(object):

    def __init__(self, filename, readCameras=True, readPoints=True, listFile=False):
        sys.stdout.write("Loading Bundle file... ")
        sys.stdout.flush()
        #print magic.from_file(filename)
        #mgc = magic.from_file(filename)

        f = open(filename,'rb');

        mgc = f.read(2)
        if [ord(i) for i in mgc] == [31, 139]:
            f.close()
            f = gzip.open(filename,'rb')

        # Bundle file v0.x
        self.version_str = f.readline()

        # n_cameras n_points
        counts = f.readline().split()
        n_cams, n_points = (int(c) for c in counts)

        # read the cameras
        if readCameras:
            sys.stdout.write("Reading cameras... ")
            sys.stdout.flush()
            self.cameras = []
            for i in range(n_cams):
                cam_lines = []
                for j in range(5): # read all 5 lines of the camera
                    cam_lines.append(f.readline())
                self.cameras.append(Camera(cam_lines))

        # read the points
        if readPoints:
            sys.stdout.write("Reading points... ")
            sys.stdout.flush()
            self.points = []
            for i in range(n_points):
                pt_lines = []
                for j in range(3):
                    pt_lines.append(f.readline())
                self.points.append(Point(pt_lines))

        f.close()

        if listFile:
            sys.stdout.write("Loading listfile...")
            sys.stdout.flush()
            self.loadListfile(listFile)

        sys.stdout.write("done.\n")
        sys.stdout.flush()


    def loadListfile(self, filename):
        f = open(filename,'r')
        self.listfile = [line.strip() for line in f]
        f.close()


class Camera(object):

    def __init__(self, cam_lines):
        # <f> <k1> <k2>
        self.focalLength, self.k1, self.k2 = (float(v) for v in cam_lines[0].split(' '))

        # <R> (3x3)
        rstring = ''.join(cam_lines[1:4])
        self.rotation = np.fromstring(rstring, sep=' ').reshape((3,3))

        # <t> (1x3)
        self.translation = np.fromstring(cam_lines[4], sep=' ')

    #TODO: radial distortion is ignored in im->world direction
    def im2cam(self, im, imWidth, imHeight):
        c = np.zeros((3,))
        c[0] = imWidth  / 2.0 - im[0] / self.focalLength
        c[1] = imHeight / 2.0 - im[1] / self.focalLength
        c[2] = 1
        return c

    def cam2world(self, c):
        #print self.rotation.transpose()
        #print c
        #print self.translation
        return self.rotation.transpose().dot(c - self.translation)

    def im2world(self, im, imWidth, imHeight):
        c = self.im2cam(im, imWidth, imHeight)
        return self.cam2world(c)

    def world2cam(self, w):
        return self.rotation.dot(w) + self.translation

    def cam2im(self, c, applyRadial, imWidth, imHeight):
        c = c / c[2]
        r = 1.0
        if applyRadial:
            cNorm2 = np.power(c[:2], 2.0).sum();
            r = 1.0 + self.k1 * cNorm2 + self.k2 * cNorm2**2
        im = np.zeros((2,))
        im[0] = imWidth  / 2.0 - r * c[0] * self.focalLength
        im[1] = imHeight / 2.0 - r * c[1] * self.focalLength
        return im


    def world2im(self, w, applyRadial, imWidth, imHeight):
        c = self.world2cam(w)
        return self.cam2im(c, applyRadial, imWidth, imHeight)


class Point(object):
    def __init__(self, pt_lines):
        # <position> (1x3)
        self.position = np.fromstring(pt_lines[0], sep=' ')

        # <color> (1x3)
        self.color = np.fromstring(pt_lines[1], sep=' ', dtype=np.uint8)

        # <view list> (n_views <v1> <v2> ...)
        view_data = pt_lines[2].split(' ')
        n_views = int(view_data[0])
        view_data = view_data[1:]
        self.views = []
        for i in range(0,n_views):
            # <camera> <key> <x> <y>
            ind = i * 4
            self.views.append(View(view_data[ind:ind+4]))


class View(object):
    def __init__(self, view_data):
        self.camera = int(view_data[0])
        self.key = int(view_data[1])
        self.x = float(view_data[2])
        self.y = float(view_data[3])



