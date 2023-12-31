import numpy


def gaussian1d(mu: float, sigma: float, x: float | numpy.ndarray) -> float | numpy.ndarray:
    exp = numpy.exp( - (x - mu)**2 / ( 2*sigma**2 ) ) 
    return ( 1 / (numpy.sqrt(2*numpy.pi)*sigma) ) * exp


def gaussian2d(
    mu_0: float, sigma_0: float, x_0: float | numpy.ndarray,
    mu_1: float, sigma_1: float, x_1: float | numpy.ndarray
) -> float | numpy.ndarray:
    exp = numpy.exp(
        - (x_0 - mu_0)**2 / ( 2*sigma_0**2 )
        - (x_1 - mu_1)**2 / ( 2*sigma_1**2 )
    ) 
    return ( 1 / (2*numpy.pi*sigma_0*sigma_1) ) * exp


