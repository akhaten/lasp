import numpy

def gaussian_filter(size: int, sigma: float, normalize: bool = False) -> numpy.ndarray:
    """Create square gaussian filter

    Params:
        - size
        - sigma
        - normalize

    Return:
        - Gaussian filter
    """

    def gaussian2d_psf(sigma: float, x: float | numpy.ndarray, y: float | numpy.ndarray) -> float | numpy.ndarray:
        exp = numpy.exp( - (x**2+y**2) / (2*sigma**2) ) 
        return ( 1 / (2*numpy.pi*sigma**2) ) * exp

    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    filter = gaussian2d_psf(sigma, x, y)
    
    if normalize:
        filter /= numpy.sum(filter)

    return filter

def north() -> numpy.ndarray:
    return numpy.array(
        [
            [0, 1, 0], 
            [0, -1, 0], 
            [0, 0, 0]
        ]
    )

def south() -> numpy.ndarray:
    return numpy.array(
        [
            [0, 0, 0], 
            [0, -1, 0], 
            [0, 1, 0]
        ]
    )

def west() -> numpy.ndarray:
    return numpy.array(
        [
            [0, 0, 0], 
            [1, -1, 0], 
            [0, 0, 0]
        ]
    )

def est() -> numpy.ndarray:
    return numpy.array(
        [
            [0, 0, 0], 
            [0, -1, 1], 
            [0, 0, 0]
        ]
    )

def laplacian() -> numpy.ndarray:
    return numpy.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]
    )


def mean_filter(size: int) -> numpy.ndarray:
    """Create mean filter

    Params:
        - size

    Return:
        - Mean filter
    """
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
