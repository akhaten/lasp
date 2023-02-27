import numpy
import numpy.linalg


# def hard(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
#     expr = numpy.abs(signal)-epsilon
#     return numpy.sign(signal) * numpy.where(0 < expr, expr, 0)


def soft(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    """sign(x) * max(|x| - epsilon, 0)
    """
    expr = numpy.abs(signal)-epsilon
    return numpy.sign(signal) * numpy.where(0 < expr, expr, 0)


def singular_value(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    """ SVT : Singular Value Thresholding
    """
    # Decomposition
    u, s, vh = numpy.linalg.svd(signal)
    # Thresholding
    singular_value_max = numpy.max(s)
    s[s < epsilon*singular_value_max] = 0.
    # Reconstruction
    res = numpy.dot(u * s, vh)
    return res


def singular_value_soft(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    """ SVT : Singular Value Thresholding
    """
    # Decomposition
    u, s, vh = numpy.linalg.svd(signal)
    # Thresholding with 
    # singular_value_max = numpy.max(s)
    s_diag = numpy.diag(s)
    s_diag = soft(s_diag, epsilon)
    s = numpy.diag(s_diag)
    # Reconstruction
    res = numpy.dot(u * s, vh)
    return res