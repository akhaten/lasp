import numpy
import numpy.linalg
import enum
from scipy.fft import fft, fft2, ifft2

import lasp.norms.vector
import lasp.thresholding
import lasp.utils



# def lasso(y: numpy.ndarray, H: numpy.ndarray, lamda: float, ro: float, nb_iterations: int) -> numpy.ndarray:
    
#     # tolerance: float
#     u = numpy.zeros_like(y)
#     x_prev = numpy.copy(y)
#     z = numpy.copy(y)

#     eigen_values = fft2(H)
#     Dh = numpy.diag(eigen_values)
    
#     # convergence = False
#     iter = 0

#     exp1 = 1 / ( Dh.T @ Dh + ro * numpy.identity(y.shape[0]) )
#     exp1 = ifft2(exp1)

#     # while not(convergence):
#     while iter < nb_iterations:

        
#         exp2 = ifft2(Dh.T) * fft2(y) + ro*z - u
#         x = ifft2(exp1) * fft2(exp2)

#         z = lasp.thresholding.soft(x, epsilon=lamda/ro)

#         u += ro * (z - x_prev)
        
#         # score = lasp.norms.vector.euclidean(x-x_prev) \
#         #     / lasp.norms.vector.euclidean(x)

#         x_prev = numpy.copy(x)
#         iter += 1

#     return x

def lasso(y: numpy.ndarray, H: numpy.ndarray, lamda: float, ro: float, nb_iterations: int) -> numpy.ndarray:

    # tolerance: float
    u_k = numpy.zeros_like(y)
    x_k = numpy.copy(y)
    z_k = numpy.copy(y)
    
    # convergence = False
    iter = 0
    
    hth = H.T @ H
    hty = H.T @ y
    n, m = hth.shape
    exp1 = numpy.linalg.inv(hth + ro * numpy.eye(n, m))

    # while not(convergence):
    while iter < nb_iterations:

        
        exp2 = hty + ro * (z_k - u_k / ro)
        x_k1 = exp1 @ exp2

        z_k1 = lasp.thresholding.soft(x_k, epsilon=lamda/ro)

        u_k1 = u_k + ro * (z_k - x_k)

        x_k = numpy.copy(x_k1)
        z_k = numpy.copy(z_k1)
        u_k = numpy.copy(u_k1)

        # score = lasp.norms.vector.euclidean(x-x_prev) \
        #     / lasp.norms.vector.euclidean(x)

        iter += 1

    return x_k

# def lasso(y: numpy.ndarray, h: numpy.ndarray, lamda: float, ro: float, nb_iterations: int) -> numpy.ndarray:

#     # tolerance: float
#     u_k = numpy.zeros_like(y)
#     x_k = numpy.copy(y)
#     z_k = numpy.copy(y)

#     n = numpy.prod(numpy.array(h.shape))
#     Dh = (1 / numpy.sqrt(n)) * fft2(h)
    
#     # convergence = False
#     iter = 0
    
#     hth = H.T @ H
#     hty = H.T @ y
#     n, m = hth.shape
#     exp1 = numpy.linalg.inv(hth + ro * numpy.eye(n, m))

#     # while not(convergence):
#     while iter < nb_iterations:

        
#         exp2 = hty + ro * (z_k - u_k / ro)
#         x_k1 = exp1 @ exp2

#         z_k1 = lasp.thresholding.soft(x_k, epsilon=lamda/ro)

#         u_k1 = u_k + ro * (z_k - x_k)

#         x_k = numpy.copy(x_k1)
#         z_k = numpy.copy(z_k1)
#         u_k = numpy.copy(u_k1)

#         # score = lasp.norms.vector.euclidean(x-x_prev) \
#         #     / lasp.norms.vector.euclidean(x)

#         iter += 1

#     return x_k

def l2(y: numpy.ndarray, h: numpy.ndarray, lamda: float) -> numpy.ndarray:

    n, m = h.shape
    h_pad = numpy.zeros_like(y)
    h_pad[0:n, 0:m] = numpy.copy(h)
    center = numpy.array(h.shape) // 2

    h_pad_shifted = lasp.utils.circshift(h_pad, 1-center)

    eigen_values = fft2(h_pad_shifted)

    d = numpy.linalg.inv(eigen_values**2 + lamda) @ eigen_values

    return ifft2(d * fft(y))


def tv(y: numpy.ndarray, H: numpy.ndarray, mu: float, ro: float, nb_iterations: int) -> numpy.ndarray:
    """Total Variation
    """
    mu * H.T @ y


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



