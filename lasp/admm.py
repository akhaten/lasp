import numpy
import numpy.linalg
from scipy.fft import fft, fft2, ifft2

import lasp.norms.vector
import lasp.thresholding

# def lasso(y: numpy.ndarray, H: numpy.ndarray, lamda: float, ro: float, nb_iterations: int) -> numpy.ndarray:
    
#     # tolerance: float
#     u = numpy.zeros_like(y)
#     x_prev = numpy.copy(y)
#     z = numpy.copy(y)

#     eigen_values = fft2(H)
#     Dh = numpy.diag(eigen_values)
    
#     # convergence = False
#     iter = 0
#     # while not(convergence):
#     while iter < nb_iterations:

#         exp1 = 1 / ( Dh.T @ Dh + ro * numpy.identity(y.shape[0]) )
#         exp2 = ifft2(Dh.T) * fft2(y) + ro*z - u
#         x = ifft2(exp1) * fft2(exp2)

#         z = lasp.utils.thresholding.soft(x, epsilon=lamda/ro)

#         u += ro * (z - x)
        
#         score = lasp.utils.norms.vector.euclidean(x-x_prev) \
#             / lasp.utils.norms.vector.euclidean(x)

#         # print(score)
#         # convergence = score < tolerance

#         x_prev = numpy.copy(x)

#         iter += 1

#     return x

def lasso(y: numpy.ndarray, H: numpy.ndarray, lamda: float, ro: float, nb_iterations: int) -> numpy.ndarray:
    pass

def tv() -> numpy.ndarray:
    """Total Variation
    """
    pass


def rpca(y: numpy.ndarray, lamda: float, mu: float, nb_iterations: int) -> numpy.ndarray:

    # tolerance: float
        
    v = numpy.zeros_like(y)
    b = numpy.copy(y) 
    t = numpy.copy(y)
    
    # convergence = False
    iter = 0

    # while not(convergence):
    while(iter < nb_iterations):
    
        b = lasp.thresholding.soft(y - t + (1/mu) * v, epsilon=lamda/mu)
        t = lasp.thresholding.singular_value_soft(y - b + (1/mu) * v, epsilon=1/mu)
        v += mu * (y - b - t)

        iter += 1

    return b, t



