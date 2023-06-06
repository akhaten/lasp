
import numpy

import lasp.metrics


def additive_white_gaussian_noise(signal: numpy.ndarray, snr: float) -> numpy.ndarray:
    
    """Additive White Gaussian Noise (AWGN)
    
    Params:
        - signal: signal
        - snr : signal to noise ratio (not in db)

    Returns:
        Signal noised
    """

    # if signal.dtype == numpy.complex64:
    #     nb_points = len(signal)
    #     power_signal = sum(numpy.abs(signal)**2) / len(signal) # puissance de l’image 
    #     power_noise = power_signal/(10**(snr/10)) # % puissance du bruit
    #     reals = numpy.random.randn(1, nb_points)
    #     imags = numpy.random.randn(1, nb_points)
    #     noise = [(numpy.sqrt(power_noise) * numpy.sqrt(1/2) * (reals[0][i] + 1j * imags[0][i])) for i in range(0, N)]# bruit Gaussien 
    #     return signal + noise

    signal_power = lasp.metrics.power(signal)
    noise_power = signal_power / snr
    sigma, mu = numpy.sqrt(noise_power), 0.0
    noise = numpy.random.normal(loc=mu, scale=sigma, size=signal.shape)
    signal_noised = signal+noise
    # grey level image must not have negative value
    signal_noised[signal_noised < 0.0] = 0.0
    return signal_noised

def multiplicative_noise(signal: numpy.ndarray, snr: float) -> numpy.ndarray:

    """Multiplicative Noise
    
    Params:
        - signal: signal
        - snr : signal to noise ratio (not in db)

    Returns:
        Signal noised
    """
    # TODO: TEST

    signal_power = lasp.metrics.power(signal)
    noise_power = signal_power / snr
    sigma, mu = numpy.sqrt(noise_power), 0.0
    noise = numpy.random.normal(loc=mu, scale=sigma, size=signal.shape)
    signal_noised = signal * (1+noise)
    # grey level image must not have negative value
    signal_noised[signal_noised < 0.0] = 0.0
    return signal_noised


def pepper_and_selt(signal: numpy.ndarray) -> numpy.ndarray:
    pass

awgn = additive_white_gaussian_noise



