""" Code from https://github.com/nanaya-tachibana """

from struct import pack
from struct import unpack
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.linalg import norm

def distortion(num, size=28):
    msb = 2051
    name = '/ais/gobi3/u/shikhar/mnist/{}-images.idx3-ubyte'
    t = 'train'
    if size != 28:
        t = t + str(size)
    filename = name.format(t)
    x = get_samples(num, filename)
    x = x.reshape((num, size, size))
    with open(filename, mode='wb') as f:
        f.write(pack('>IIII', msb, num, size, size))
        for i in range(num):
            d = np.array(np.floor(elastic_distortion(x[i])), dtype=np.uint8)
            f.write(pack('B'*size**2, *d.flatten()))

def read_images(f, row, col, num):
    pixels = row * col
    buf = f.read(pixels*num)
    return [
        unpack('B'*pixels, buf[p:p+pixels])
        for p in np.arange(0, num*pixels, pixels)
    ]

def get_samples(num, f):
    with open(f, mode='rb') as images:
        images.seek(4)
        # read total number of images and number of row pixels and colum pixels
        _num, row, col = unpack('>III', images.read(12))
        if num > _num:
            num = _num
        img = read_images(images, row, col, num)
        return np.array(img, dtype=np.uint8)

def elastic_distortion(image, sigma=6, alpha=36):
    """The generation of the elastic distortion.
    First, random displacement fields are created from a uniform distribution
    between -1 and +1. They are then convolved with a Gaussian of standard
    deviation sigma. After normalization and multiplication by a scaling
    factor alpha that controls the intensity of the deformation, they are
    applied on the image. sigma stands for the elastic coefficient. A small
    sigma means more elastic distortion. For a large sigma, the deformation
    approaches affine, and if sigma is very large, then the displacements
    become translations.
    """
    def delta():
        d = gaussian_filter(np.random.uniform(-1, 1, size=image.shape), sigma)
        return (d / norm(d)) * alpha

    assert image.ndim == 2
    dx = delta()
    dy = delta()
    return bilinear_interpolate(image, dx, dy)


def bilinear_interpolate(values, dx, dy):
    """Interpolating with given dx and dy"""
    assert values.shape == dx.shape == dy.shape

    A = np.zeros(values.shape)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            x = i + dx[i, j]
            y = j + dy[i, j]
            if x < 0:
                x = x + int(1 + 0 - x)
            if x >= values.shape[0] - 1:
                x = x - int(1 + x - (values.shape[0] - 1))
            if y < 0:
                y = y + int(1 + 0 - y)
            if y >= values.shape[1] - 1:
                y = y - int(1 + y - (values.shape[1] - 1))

            x1 = int(x)
            y1 = int(y)
            x2 = x1 + 1
            y2 = y1 + 1
            f11 = values[x1, y1]
            f12 = values[x1, y2]
            f21 = values[x2, y1]
            f22 = values[x2, y2]

            A[i, j] = (
                f11*(x2-x)*(y2-y) + f12*(x2-x)*(y-y1)
                + f21*(x-x1)*(y2-y) + f22*(x-x1)*(y-y1)
            )
    return A

if __name__ == '__main__':
    distortion(60000)

