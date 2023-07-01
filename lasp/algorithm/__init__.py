import numpy
import numpy.linalg
import scipy.signal
import enum


def van_cittert(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, nb_iterations: int) -> numpy.ndarray:
    #TODO : TEST
    u = numpy.copy(image_fft2)
    g = 1-kernel_fft2
    for _ in range(0, nb_iterations):
        u = image_fft2 + g * u
    return u  


def secb(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, ks: float, s: float) -> numpy.ndarray:
    #TODO : TEST
    
    e1 = numpy.conj(kernel_fft2)
    
    e2 = numpy.absolute(kernel_fft2)**2
    
    I = numpy.ones_like(kernel_fft2)
    kernel_fft2_s = kernel_fft2 ** s
    e3 = I - kernel_fft2_s
    
    e4 = e2 + ks * numpy.conj(e3) * e3

    e5 = e1 / e4

    return e5 * image_fft2


def richardson_lucy_algorithm(image: numpy.ndarray, kernel: numpy.ndarray, nb_iterations: int) -> numpy.ndarray:
    #TODO : TEST

    res = numpy.copy(image)
    # kernel with columns reversed
    ker = numpy.flip(kernel, 1)
    
    for _ in range(0, nb_iterations):
        conv1 = scipy.signal.convolve2d(res, kernel, mode='same')
        e1 = image / conv1
        conv2 = scipy.signal.convolve2d(e1, ker, same='same')
        res = res * conv2
    
    return res


def blind_deconvolution_richardson_lucy_algorithm(image: numpy.ndarray, nb_iterations: int, kernel_init: numpy.ndarray = None) -> numpy.ndarray:
    
    #TODO : TEST

    res = numpy.copy(image)

    kernel = numpy.zeros_like(image) if kernel_init is None else kernel_init    
    
    for _ in range(0, nb_iterations):

        # Estimation of kernel
        res_hat = numpy.flip(res, 1) # res with columns reversed
        kernel = (kernel / numpy.sum(res)) * ( \
                    scipy.signal.convolve2d( \
                        image / scipy.signal.convolve2d(res, kernel, 'same'), res_hat, 'same')
                )
        
        # Estimation of image res
        kernel_hat = numpy.flip(kernel, 1) # kernel with columns reversed
        res = res * ( \
                    scipy.signal.convolve2d( \
                        image / scipy.signal.convolve2d(res, kernel, 'same'), kernel_hat, 'same')
                )

    return res