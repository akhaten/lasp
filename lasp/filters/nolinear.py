import numpy
# import numpy.fft
import scipy.signal
import enum

import lasp.filters.linear

def bilateral(image: numpy.ndarray, sigma_spatial: float, sigma_color: float, size: int) -> numpy.ndarray:

    """ Bilateral filter
        
        Parameters:
            - image: image
            - sigma_d: sigma spatial
            - sigma_r: sigma color
            - size: size of window/neighbourhood

        Returns:
            - image filtered
    """

    def weight(i: int, j: int, k: int, l: int) -> float:
        expr1: float = ( (i-k)**2 + (j-l) ** 2 ) / ( 2*(sigma_spatial**2) )
        expr2: float = ( (image[i, j]-image[k, l]) ** 2 ) / ( 2*(sigma_color**2) )
        return numpy.exp(-expr1-expr2)

    nb_rows, nb_cols = image.shape
    half_size = size // 2

    image_d = numpy.copy(image)
    
    for i in range(half_size, nb_rows-half_size):
        for j in range(half_size, nb_cols-half_size):
            sum_iw = 0.0
            sum_w = 0.0
            for k in range(i-half_size, i+half_size):
                for l in range(j-half_size, j+half_size):
                    w = weight(i, j, k, l)
                    sum_iw += image[k, l] * w
                    sum_w += w

            image_d[i, j] = sum_iw / sum_w
   
    return image_d


def non_local_mean(image: numpy.ndarray, sigma: float, size: int) -> numpy.ndarray:

    def weight(i: int, j: int, k: int, l: int) -> float:
        expr: float = ((image[i, j]-image[k, l]) ** 2) / (2*(sigma**2))
        return numpy.exp(-expr)

    nb_rows, nb_cols = image.shape
    half_size = size // 2

    image_d = numpy.copy(image)
    
    for i in range(half_size, nb_rows-half_size):
        for j in range(half_size, nb_cols-half_size):
            sum_iw = 0.0
            sum_w = 0.0
            for k in range(i-half_size, i+half_size):
                for l in range(j-half_size, j+half_size):
                    w = weight(i, j, k, l)
                    sum_iw += image[k, l] * w
                    sum_w += w

            image_d[i, j] = sum_iw / sum_w
   
    return image_d


class Mask(enum.Enum):
    NORTH = [[0, 1, 0], [0, -1, 0], [0, 0, 0]]
    SOUTH = [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
    WEST  = [[0, 0, 0], [1, -1, 0], [0, 0, 0]]
    EST   = [[0, 0, 0], [0, -1, 1], [0, 0, 0]]

def anisotropic(image: numpy.ndarray, lamda: float, k: float, nb_iterations: int) -> numpy.ndarray:

    # M1 = compute(us, n=20, k=30, lamb=0.1)

    # res = numpy.abs(scipy.signal.hilbert2(M1))
    # figure = matplotlib.pyplot.figure(figsize=(50, 50), dpi=20)
    # # matplotlib.pyplot.subplot(1, 2, 1)
    # matplotlib.pyplot.imshow(numpy.log(res), cmap='gray', aspect='auto')

    """
        k  [0, 20]
        lamb ]0, 25]
        n ~ 20
    """

    class Mask(enum.Enum):
        NORTH = list(lasp.filters.linear.north())
        SOUTH = list(lasp.filters.linear.south())
        WEST  = list(lasp.filters.linear.west())
        EST   = list(lasp.filters.linear.est())


    c = lambda u : numpy.exp(-(u/k)**2)
    image_n = numpy.copy(image)
    masks = list(Mask)

    for i in range(0, nb_iterations):

        grad = {}
        c_dir = {}
        for mask in masks:
            grad[mask] = scipy.signal.convolve2d(image_n, mask.value, mode = 'same')
            c_dir[mask] = c(grad[mask])

        image_n = image_n + lamda * (
              grad[Mask.NORTH] * c_dir[Mask.NORTH]
            + grad[Mask.EST]   * c_dir[Mask.EST]
            + grad[Mask.WEST]  * c_dir[Mask.WEST]
            + grad[Mask.SOUTH] * c_dir[Mask.SOUTH]
        )

    return image_n


def mean_geometric(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, k: float, s: float) -> numpy.ndarray:
    #TODO : TEST
    e1 = 1 / ( numpy.absolute(kernel_fft2) ** s )
    e2 = numpy.conj(kernel_fft2) / (numpy.absolute(kernel_fft2)**2 + k)
    e2 = e2 ** (1-s)
    e3 = e1 * e2
    return e3 * image_fft2


def wiener(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, k: float) -> numpy.ndarray:
    frac = numpy.conj(kernel_fft2) / (numpy.absolute(kernel_fft2)**2 + k)
    return frac * image_fft2

def inverse(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray) -> numpy.ndarray:
    return image_fft2 / kernel_fft2


def pseudo_inverse(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, threshold: float) -> numpy.ndarray:
    return numpy.where(
        threshold <= numpy.abs(kernel_fft2), 
        image_fft2 / kernel_fft2, 
        image_fft2
    )