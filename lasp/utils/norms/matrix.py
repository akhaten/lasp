import numpy
import scipy.linalg

def norm2(matrix: numpy.ndarray) -> float:
    return numpy.max(scipy.linalg.svdvals(matrix))