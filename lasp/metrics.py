import numpy

def mean_absolute_error(signal_1: numpy.ndarray, signal_2: numpy.ndarray) -> float:
    """ Mean Absolute Error (MAE)
    """
    n = signal_1.shape[0] * signal_1.shape[1]
    # | signal_1[i] - signal_2[i] |
    tmp = numpy.abs(signal_1 - signal_2)
    # sum | signal_1[i] - signal_2[i] |
    res = numpy.sum(tmp)
    # (1/n) sum | signal_1[i] - signal_2[i] |
    res /= n
    return res


def mean_squared_error(signal_1: numpy.ndarray, signal_2: numpy.ndarray) -> float:
    """ Mean Squared Error (MSE)
    """
    n = signal_1.shape[0] * signal_1.shape[1]
    # signal_1[i] - signal_2[i]
    tmp = numpy.power(signal_1 - signal_2, 2)
    # (signal_1[i] - signal_2[i])^2
    # tmp **= 2
    # sum ( (signal_1[i] - signal_2[i])^2 )
    res = numpy.sum(tmp)
    # (1/n) sum ( (signal_1[i] - signal_2[i])^2 )
    res /= n
    return res

def peak_signal_to_noise_ratio(signal_1: numpy.ndarray, signal_2: numpy.ndarray, intensity_max: float = 255) -> float:
    """ Peak Signal to Noise Ratio (PSNR)
    """
    # intensity_max = numpy.max(signal_1)
    mse = mean_squared_error(signal_1, signal_2)
    return 10 * numpy.log10( (intensity_max**2) / mse )


def structural_similarity_index_measure(signal_1: numpy.ndarray, signal_2: numpy.ndarray, k1: float = 0.01, k2: float = 0.03) -> float:
    """ Structural Similarity Index Measure (SSIM)
    """

    #TODO : TEST
    L = 255

    c1 = (k1*L)**2
    c2 = (k2*L)**2
    c3 = c2 / 2


    mean1 = numpy.mean(signal_1)
    var1 = numpy.var(signal_1)
    mean2 = numpy.mean(signal_2)
    var2 = numpy.var(signal_2)
    cov = numpy.cov(signal_1, signal_2)

    l = ( 2*mean1*mean2 + c1) / ( mean1**2 + mean2**2 + c1 )
    c = ( 2*var1*var2 + c2 ) / ( var1**2 + var2**2 + c2 )
    s = ( cov + c3 ) / ( var1*var2 + c3 )

    return l*c*s

def structural_disimilarity_index_measure(signal_1: numpy.ndarray, signal_2: numpy.ndarray, k1: float = 0.01, k2: float = 0.03) -> float:
    """ Structural Disimilarity Index Measure (DSSIM)
    """

    #TODO : TEST
    return (1 - SSIM(signal_1, signal_2, k1, k2)) / 2


def power(signal: numpy.ndarray) -> float:
    """Power of signal
    """
    # nb_value = numpy.prod(numpy.array(signal.shape))
    # sum = numpy.sum(numpy.power(signal, 2))
    # return sum / nb_value
    return numpy.mean(numpy.power(signal, 2))





# def power_noise(power_signal: float, snr: float) -> float:
#     return power_signal / ( 10 ** (snr/10) )



MAE = mean_absolute_error
MSE = mean_squared_error
PSNR = peak_signal_to_noise_ratio
SSIM = structural_similarity_index_measure
DSSIM = structural_disimilarity_index_measure



