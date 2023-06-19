import numpy
import numpy.linalg


def least_square(A: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    """Least square estimator
    
    Solve Ax = b.

    Params:
        - A
        - b

    Return:
        - solution x 
    """
    return numpy.linalg.inv(A.T @ A) @ A.T @ b


