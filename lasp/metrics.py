import numpy

def mean_absolute_error(signal_1: numpy.ndarray, signal_2: numpy.ndarray) -> float:
    n = signal_1.shape[0] * signal_1.shape[1]
    # | signal_1[i] - signal_2[i] |
    tmp = numpy.abs(signal_1 - signal_2)
    # sum | signal_1[i] - signal_2[i] |
    res = numpy.sum(tmp)
    # (1/n) sum | signal_1[i] - signal_2[i] |
    res /= n
    return res


def mean_squared_error(signal_1: numpy.ndarray, signal_2: numpy.ndarray) -> float:
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

def peak_signal_to_noise_ratio(signal_1: numpy.ndarray, signal_2: numpy.ndarray) -> float:
    intensity_max = numpy.max(signal_1)
    mae = mean_squared_error(signal_1, signal_2)
    return 10 * numpy.log10( (intensity_max**2) / mae )

def power(signal: numpy.ndarray) -> float:
    # nb_value = numpy.prod(numpy.array(signal.shape))
    # sum = numpy.sum(numpy.power(signal, 2))
    # return sum / nb_value
    return numpy.mean(numpy.power(signal, 2))



# def power_noise(power_signal: float, snr: float) -> float:
#     return power_signal / ( 10 ** (snr/10) )



# mae = mean_absolute_error
# psnr = peak_signal_to_noise_ratio
# snr = signal_to_noise_ratio
# snr_db = signal_to_noise_ratio_db


