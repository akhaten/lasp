import numpy
import numpy.linalg


def least_square(A: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    return numpy.linalg.inv(A.T @ A) @ A.T @ b


