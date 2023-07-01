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

def multidimensional_soft(d: numpy.ndarray, epsilon: float):
    """ Thresholding soft for multidimensional array
    Use generalization of sign function
    
    Params:
        - d : multidimensional array
        - epsilon : threshold

    Return:
        Array thresholded with dimesion equal to d
    """
    s = numpy.sqrt(numpy.sum(d**2, axis=0))
    ss = numpy.where(s > epsilon, (s-epsilon)/s, 0)
    output = numpy.array([ss*d[i] for i in range(0, d.shape[0])])
    return output


def singular_value(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    """ SVT : Singular Value Thresholding

    Params:
        - d : multidimensional array
        - epsilon : threshold

    Return:
        Array thresholded with dimesion equal to d
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
    """ SVT : Singular Value Soft Thresholding
    Apply soft threshoding on singular values of signal

    Params:
        - d : multidimensional array
        - epsilon : threshold

    Return:
        Array thresholded with dimesion equal to d
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