import numpy
import numpy.linalg
import scipy.signal
import enum



def bilateral_filter(image: numpy.ndarray, sigma_spatial: float, sigma_color: float, size: int) -> numpy.ndarray:

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


def gaussian_filter(size: int, sigma: float, normalize: bool = False) -> numpy.ndarray:

    def gaussian2d_psf(sigma: float, x: float | numpy.ndarray, y: float | numpy.ndarray) -> float | numpy.ndarray:
        exp = numpy.exp( - (x**2+y**2) / (2*sigma**2) ) 
        return ( 1 / (2*numpy.pi*sigma**2) ) * exp


    # half_size = size // 2
    # filter = numpy.zeros((size, size))
    # for x in range(-half_size, half_size+1):
    #     for y in range(-half_size, half_size+1):
    #         filter[half_size+x, half_size+y] = gaussian2d_psf(sigma, x, y)

    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    filter = gaussian2d_psf(sigma, x, y)
    
    if normalize:
        filter /= numpy.sum(filter)

    return filter

# mask_n = numpy.array(
#     [
#         [0, 1, 0],
#         [0, -1, 0],
#         [0, 0, 0]
#     ]
# )

# mask_s = numpy.array(
#     [
#         [0, 0, 0],
#         [0, -1, 0],
#         [0, 1, 0]
#     ]
# )

# mask_w = numpy.array(
#     [
#         [0, 0, 0],
#         [1, -1, 0],
#         [0, 0, 0]
#     ]
# )

# mask_e = numpy.array(
#     [
#         [0, 0, 0],
#         [0, -1, 1],
#         [0, 0, 0]
#     ]
# )

# class Mask(enum.Enum):
#     NORTH=numpy.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
#     SOUTH=numpy.array([[0, 0, 0],[0, -1, 0],[0, 1, 0]])
#     WEST = numpy.array([[0, 0, 0],[1, -1, 0],[0, 0, 0]])
#     EST = numpy.array([[0, 0, 0],[0, -1, 1],[0, 0, 0]])

class Mask(enum.Enum):
    NORTH = [[0, 1, 0], [0, -1, 0], [0, 0, 0]]
    SOUTH = [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
    WEST  = [[0, 0, 0], [1, -1, 0], [0, 0, 0]]
    EST   = [[0, 0, 0], [0, -1, 1], [0, 0, 0]]

def anisotropic_filter(image: numpy.ndarray, lamda: float, k: float, nb_iterations: int) -> numpy.ndarray:

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
            + grad[Mask.WEST]  * c_dir[Mask.EST]
            + grad[Mask.SOUTH] * c_dir[Mask.SOUTH]
        )


        # grad_n = scipy.signal.convolve2d(image_n, mask_n, mode = 'same')
        # grad_e = scipy.signal.convolve2d(image_n, mask_e, mode = 'same')
        # grad_w = scipy.signal.convolve2d(image_n, mask_w, mode = 'same')
        # grad_s = scipy.signal.convolve2d(image_n, mask_s, mode = 'same')

        # c_n, c_e, c_w, c_s = c(grad_n), c(grad_e), c(grad_w), c(grad_s)

        # image_n = image_n + lamda * (grad_n*c_n+grad_s*c_s+grad_w*c_w+grad_e*c_e)

    return image_n




    
def mean_geometric(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, k: float, s: float) -> numpy.ndarray:
    #TODO : TEST
    e1 = 1 / ( numpy.absolute(kernel_fft2) ** s )
    e2 = numpy.conj(kernel_fft2) / (numpy.absolute(kernel_fft2)**2 + k)
    e2 = e2 ** (1-s)
    e3 = e1 * e2
    return e3 * image_fft2

# def wiener_filter(h_fft2: numpy.ndarray, k: float) -> numpy.ndarray:
#     return numpy.conj(h_fft2) / (numpy.absolute(h_fft2)**2 + k)

# def wiener_method(i_fft2: numpy.ndarray, h_fft2: numpy.ndarray, k: float) -> numpy.ndarray:
#     wiener: numpy.ndarray = wiener_filter(h_fft2, k)
#     return i_fft2 * wiener

def inverse(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray) -> numpy.ndarray:
    return image_fft2 / kernel_fft2

def pseudo_inverse(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, threshold: float) -> numpy.ndarray:
    return numpy.where(
        threshold <= numpy.abs(kernel_fft2), 
        image_fft2 / kernel_fft2, 
        image_fft2
    )

def van_cittert(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, nb_iterations: int) -> numpy.ndarray:
    #TODO : TEST
    u = numpy.copy(image_fft2)
    g = 1-kernel_fft2
    for _ in range(0, nb_iterations):
        u = image_fft2 + g * u
    return u  





def mean_filter(size: int) -> numpy.ndarray:
    filter = numpy.ones(shape=(size, size))
    filter /= size*size
    return filter


def roberts_masks() -> numpy.ndarray:
    #TODO : TEST
 
    return numpy.array(
        [
            numpy.array([[-1, 0], [0, 1]]),
            numpy.array([[0, -1], [1, 0]])
        ]
    )

def sobel_masks() -> numpy.ndarray:
    #TODO : TEST
 
    return numpy.array(
        [
            numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        ]
    )

def kirsh_masks() -> numpy.ndarray:
    #TODO
    # return numpy.array(
    #     [
    #         numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    #         numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #     ]
    # )
    pass

def robinson_masks() -> numpy.ndarray:
    #TODO
    # return numpy.array(
    #     [
    #         numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    #         numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #     ]
    # )
    pass


# Deconvolution

def wiener(image_fft2: numpy.ndarray, kernel_fft2: numpy.ndarray, k: float) -> numpy.ndarray:
    frac = numpy.conj(kernel_fft2) / (numpy.absolute(kernel_fft2)**2 + k)
    return frac * image_fft2

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
    
