import numpy
import matplotlib.pyplot


def histogram(image_greyscale: numpy.ndarray) -> numpy.ndarray:
    m, n = image_greyscale.shape
    hist = numpy.zeros(shape=256)
    for i in range(0, m):
        for j in range(0, n):
            hist[image_greyscale[m, n]] += 1
    return hist


# def recadrage(image_greyscale: numpy.ndarray) -> numpy.ndarray:

#     def t(f) -> float:
        
