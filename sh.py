import numpy as np
import scipy
from scipy.special import sph_harm

class SHT:
    def __init__(self, L, az_samples, zen_samples):
        self.az_samples = az_samples
        self.zen_samples = zen_samples
        self.theta = np.linspace(0, 2*np.pi, num=self.az_samples, endpoint=False)
        self.phi = np.linspace(0, np.pi, num=self.zen_samples, endpoint=True)
        self.Yc = np.zeros((L+1,2*L+1,az_samples,zen_samples), dtype='complex128')
        for l in range(L+1):
            for m in range(-l, l+1):
                for t in range(self.theta.size):
                    for p in range(self.phi.size):
                        self.Yc[l,m,t,p] = sph_harm(m, l, self.theta[t], self.phi[p])

    def _compute_reals(self):
        self.Ylm = np.zeros_like(self.Yc, dtype='double')
        for l in range(self.Ylm.shape[0]):
            for m in range(-l,l+1):
                if m == 0:
                    values = self.Yc[l,m,:,:]
                elif m < 0:
                    values = (1j / np.sqrt(2)) * (self.Yc[l,m,:,:] - (-1)**m * self.Yc[l,-m,:,:])
                else:  # m > 0
                    values = (1.0 / np.sqrt(2)) * (self.Yc[l,-m,:,:] + (-1)**m * self.Yc[l,m,:,:])

                #pipi.imshow(np.abs(np.imag(values)))
                self.Ylm[l,m,:,:] = np.real(values)
                #pipi.imshow(self.Ylm[l,m,:,:])

    # likely broken - scaling issue, I think
    def sht(self, f, L):
        if f.shape != self.Ylm.shape[2:]:
            print "wrong shape for sht"
            return None

        #dph = 1.0 / self.zen_samples
        #dth = 1.0 / self.az_samples
        areas = np.ones(f.shape) * np.sin(self.phi)[np.newaxis,:]
        areas = areas / np.sum(areas)

        coeffs = []
        for l in range(L+1):
            for m in range(-l,l+1):
                coeffs.append(np.sum(f * areas * self.Ylm[l,m,:,:]))

        return coeffs

    def isht(self, c, L):
        f = np.zeros((self.az_samples, self.zen_samples))
        i = 0
        for l in range(L+1):
            for m in range(-l,l+1):
                f += c[i] * self.Ylm[l,m,:,:]
                i += 1

        return f


if __name__ == '__main__':
    import pipi
    import pipi.sh

    sht = pipi.sh.SHT(1, 64, 32)
    sht._compute_reals()

    imp = np.zeros((sht.az_samples, sht.zen_samples))
    imp[32:,:] = 1

    c = sht.sht(imp,1)
    f = sht.isht(c,1)

    pipi.imshow(imp)
    pipi.imshow(f)
