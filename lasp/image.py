import numpy
import matplotlib.pyplot


def histogram(image_greyscale: numpy.ndarray) -> numpy.ndarray:
    m, n = image_greyscale.shape
    hist = numpy.zeros(shape=256)
    for i in range(0, m):
        for j in range(0, n):
            hist[image_greyscale[m, n]] += 1
    return hist


def sparse(shape: tuple[int, int], epsilon: float) -> numpy.ndarray:
    m, n = shape
    nb_pixel_to_one = int((1-epsilon)*m*n)
    d: numpy.ndarray = numpy.zeros(shape=m*n)
    d[0:nb_pixel_to_one] = 1
    d = numpy.random.permutation(d)
    return numpy.reshape(d, (m, n))


