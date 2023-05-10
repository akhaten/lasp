import numpy
import numpy.linalg
import enum
from scipy.fft import fft, fft2, ifft2

import lasp.norms.vector
import lasp.thresholding
import lasp.utils


def lasso(y: numpy.ndarray, h: numpy.ndarray, lamda: float, ro: float, nb_iterations: int) -> numpy.ndarray:
    
    n, m = h.shape
    h_pad = numpy.zeros_like(y)
    h_pad[0:n, 0:m] = numpy.copy(h)
    center = numpy.array([numpy.round(n/2), numpy.round(m/2)], dtype=int)

    h_pad_shifted = lasp.utils.circshift(h_pad, 1-center)

    eigen_values = numpy.array(fft2(h_pad_shifted))

    x = numpy.copy(y)
    z = numpy.copy(y)
    u = numpy.zeros_like(y)

    expr1 = 1 / ( eigen_values.conj() * eigen_values + ro )
    expr2 = ifft2(eigen_values * fft2(y))

    for _ in range(0, nb_iterations):

        x = numpy.real(ifft2(expr1 * fft2(expr2 + ro*z - u)))
        z = lasp.thresholding.soft(x + u / ro, lamda / ro)
        u = u + ro * (x - z)

    return x


# def FTHF(h: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:

#     n, m = h.shape
#     h_pad = numpy.zeros_like(y)
#     h_pad[0:n, 0:m] = numpy.copy(h)
#     center = numpy.array(h.shape) // 2

#     h_pad_shifted = lasp.utils.circshift(h_pad, 1-center)
#     bccb_eigen_values = fft2(h_pad_shifted)

#     return ifft2(bccb_eigen_values * fft2(y))

def tikhonov(y: numpy.ndarray, h: numpy.ndarray, lamda: float) -> numpy.ndarray:

    n, m = h.shape
    h_pad = numpy.zeros_like(y)
    h_pad[0:n, 0:m] = numpy.copy(h)
    
    center = numpy.array([numpy.round(n/2), numpy.round(m/2)], dtype=int)


    h_pad_shifted = lasp.utils.circshift(h_pad, 1-center)

    # Eigens values of patrix bccb generate by h
    bccb_eigen_values = numpy.array(fft2(h_pad_shifted))

    inv = ( 1 / (bccb_eigen_values.conj() * bccb_eigen_values + lamda) )
    d = inv * bccb_eigen_values.conj()

    return numpy.real(ifft2(d * fft2(y)))

def l2(y: numpy.ndarray, h: numpy.ndarray, lamda: float) -> numpy.ndarray:
    return tikhonov(y, h, lamda)


# def gcv(y: numpy.ndarray, h: numpy.ndarray, lamda: float) -> float:


#     # L2 
#     n, m = h.shape
#     h_pad = numpy.zeros_like(y)
#     h_pad[0:n, 0:m] = numpy.copy(h)
#     center = numpy.array(h.shape) // 2

#     h_pad_shifted = lasp.utils.circshift(h_pad, 1-center)

#     # Eigens values of matrix bccb generate by h
#     eigen_values = fft2(h_pad_shifted)

#     f_lamda = ( 1 / (eigen_values**2 + lamda) ) @ eigen_values

#     Hf_lamda = ifft2(f_lamda * fft2(f_lamda))
#     x_estim = ifft2(f_lamda * fft(y))
#     Hx_estim = ifft2(f_lamda * fft(x_estim))


#     identity = numpy.eye(y.shape[0], y.shape[1])

#     expr1 = numpy.sum((identity - Hx_estim)**2)
#     expr2 = numpy.trace(identity-)

#     exp2

    




#     f_lamda = l2(y, h, lamda)


    





# def tv(y: numpy.ndarray, h: numpy.ndarray, lamda: float, ro: float, nb_iterations: int) -> numpy.ndarray:
#     """Total Variation
#     """

#     def D(n: int) -> numpy.ndarray:
#         d = numpy.zeros((n, n+1))
#         identity = numpy.eye(n, n)
#         d[0:n, 0:n] -= identity
#         d[0:n, 1:n+1] += identity
#         return d

    

#     y_lexico = numpy.reshape(y, newshape=(-1, 1), order='F')
#     n, _ = y_lexico.shape

#     d = D(n)



#     mu * H.T @ y


def rpca(y: numpy.ndarray, lamda: float, mu: float, nb_iterations: int) -> numpy.ndarray:

    # tolerance: float
        
    v = numpy.zeros_like(y)
    b = numpy.copy(y) 
    t = numpy.copy(y)
    
    # convergence = False
    iter = 0

    # while not(convergence):
    for _ in range(0, nb_iterations):
    
        b = lasp.thresholding.soft(y - t + (1/mu) * v, epsilon=lamda/mu)
        t = lasp.thresholding.singular_value_soft(y - b + (1/mu) * v, epsilon=1/mu)
        v += mu * (y - b - t)


    return b, t



