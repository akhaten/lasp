import numpy
import numpy.linalg
import enum
from scipy.fft import fft, fft2, ifft2

import lasp.norms.vector
import lasp.thresholding
import lasp.utils
import lasp.differential
import lasp.filters.linear


def tikhonov(y: numpy.ndarray, h: numpy.ndarray, lamda: float) -> numpy.ndarray:
    """Tikhonov Regularization
    """

    h_diag = lasp.utils.fourier_diagonalization(
        kernel = h,
        shape_out = y.shape
    )

    h2_diag = numpy.abs(h_diag)**2

    tmp = (numpy.conj(h_diag) * numpy.fft.fft2(y)) / (h2_diag + 2*lamda)
    
    return numpy.real(numpy.fft.ifft2(tmp))


def lasso(y: numpy.ndarray, h: numpy.ndarray, lamda: float, ro: float, nb_iterations: int) -> numpy.ndarray:
    """ L1 regularization with ADMM
    """
    
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

    for i in range(0, nb_iterations):

        # print('iter {}'.format(i))

        x = numpy.real(ifft2(expr1 * fft2(expr2 + ro*z - u)))
        # print('x_min : {} \t x_max : {}\n'.format())
        z = lasp.thresholding.soft(x + u / ro, lamda / ro)
        u = u + ro * (x - z)

    return x


def tv(
    y: numpy.ndarray, 
    h: numpy.ndarray, 
    lamda: float, 
    sigma: float, 
    nb_iterations: int,
    tol: float = 0
) -> numpy.ndarray:
    """Total Variation Regularization with Split Bregman
    """
    
    lap_diag = lasp.utils.fourier_diagonalization(
        kernel=lasp.filters.linear.laplacian(),
        shape_out=y.shape
    )

    h_diag = lasp.utils.fourier_diagonalization(
        kernel=h,
        shape_out=y.shape
    )

    h2_diag = numpy.abs(h_diag)**2


    cst1 = h2_diag + sigma*lap_diag
    cst2 = numpy.conj(h_diag)*numpy.fft.fft2(y)
   

    # INitialization
    u = numpy.copy(y) 
    d_x=numpy.zeros_like(y)
    d_y=numpy.zeros_like(y)
    b_x=numpy.zeros_like(y)
    b_y=numpy.zeros_like(y)

    for i in range(0, nb_iterations):

        a = sigma * (
            lasp.differential.dxT(d_x-b_x)
            + lasp.differential.dyT(d_y-b_y)
        )

        b = numpy.fft.fft2(a) + cst2

        u0 = numpy.copy(u)
        
        u = numpy.real(numpy.fft.ifft2(b / cst1))

        err = numpy.linalg.norm(u-u0, 'fro') \
            / numpy.linalg.norm(u, 'fro')

        if i%10 == 0:
            print('Iterations: {} ! \t error is: {}'.format(i, err))

        if err <= tol:
            break

        d_x, d_y = lasp.thresholding.multidimensional_soft(
            numpy.array(
                [
                    lasp.differential.dx(u)+b_x,
                    lasp.differential.dy(u)+b_y
                ]
            ),
            lamda/sigma
        )

        b_x=b_x+lasp.differential.dx(u)-d_x
        b_y=b_y+lasp.differential.dy(u)-d_y

    min_u = numpy.min(u)
    max_u = numpy.max(u)
    u = (u-min_u) / (max_u-min_u)

    return u


def rpca(y: numpy.ndarray, lamda: float, mu: float, nb_iterations: int) -> numpy.ndarray:
    """Robust Principal Components Analysis
    """
        
    v = numpy.zeros_like(y)
    b = numpy.copy(y) 
    t = numpy.copy(y)
    
    for _ in range(0, nb_iterations):
    
        b = lasp.thresholding.soft(y - t + (1/mu) * v, epsilon=lamda/mu)
        t = lasp.thresholding.singular_value_soft(y - b + (1/mu) * v, epsilon=1/mu)
        v += mu * (y - b - t)

    return b, t



