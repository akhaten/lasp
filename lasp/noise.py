
import numpy

import lasp.metrics


def additive_gaussian_noise(signal: numpy.ndarray, snr: float) -> numpy.ndarray:
    """awgn
        signal: signal
        snr : signal to noise ratio (not in db)
    """
    signal_power = lasp.metrics.power(signal)
    noise_power = signal_power / snr
    sigma, mu = numpy.sqrt(noise_power), 0.0
    noise = numpy.random.normal(loc=mu, scale=sigma, size=signal.shape)
    signal_noised = signal+noise
    # grey level image must not negative value
    signal_noised[signal_noised < 0.0] = 0.0
    return signal_noised

# def additive_gaussian_noise(signal: numpy.ndarray, snr: float) -> numpy.ndarray:
#     """
#         signal: signal
#         snr : signal to noise ratio (not in db)
#     """

#     p_signal = lasp.utils.metrics.power(signal)
#     p_normalized = p_signal / snr
    
#     noise = numpy.random.randn(numpy.prod(signal.shape))
#     noise = numpy.reshape(noise, signal.shape)

#     p_noise = lasp.utils.metrics.power(noise)
#     # Normal(mu, sigma^2) = > sigma * np.random.randn(...) + mu
#     noise *= ( numpy.sqrt(p_noise) / p_normalized)

#     noised = signal+noise
    
#     return noised

# def additive_gaussian_noise(signal: numpy.ndarray, snr: float = 40.) -> numpy.ndarray:
    
#     # SNR = 40; 
#     # power_signal = numpy.sum(signal**2) # puissance du signal
#     # power_noise = power_signal / (10**(snr/10)) # puissance du bruit 
#     # mu = 0
#     # sigma = numpy.sqrt(power_noise)
#     # gaussian_noise = numpy.random.normal(loc=mu, scale=sigma, size=signal.shape)
#     # return signal + gaussian_noise
    
#     power_signal = sum(sum(signal**2)) # puissance de l’image 
#     power_noise_norm = power_signal/(10**(snr/10)) # % puissance du bruit 
#     # print(power_noise_norm)
#     noise=numpy.random.randn(signal.shape[0], signal.shape[1]) # bruit Gaussien 
#     noise_norm = noise/numpy.sqrt(sum(sum(noise**2)))*numpy.sqrt(power_noise_norm) # régler la puissance du bruit 
#     return signal + noise_norm # bruiter le signal


# # def additive_gaussian_noise(signal: numpy.ndarray, snr: float = 10.) -> numpy.ndarray:
# #     nb_points = len(signal)
# #     power_signal = sum(numpy.abs(signal)**2) / len(signal) # puissance de l’image 
# #     power_noise = power_signal/(10**(snr/10)) # % puissance du bruit
# #     reals = numpy.random.randn(1, nb_points)
# #     imags = numpy.random.randn(1, nb_points)
# #     noise = [(numpy.sqrt(power_noise) * numpy.sqrt(1/2) * (reals[0][i] + 1j * imags[0][i])) for i in range(0, N)]# bruit Gaussien 
# #     return signal + noise


