import numpy

def signal_to_noise_ratio(signal_power: float, noise_power: float) -> float:
    """Compute Signal to Noise Ratio (SNR)
    
    Params:
        signal_power : power of signal
        noise_power : power of noise
    
    Return:
        - snr
    """
    return signal_power / noise_power

def signal_to_noise_ratio_db(signal_power: float, noise_power: float) -> float:
    """Compute Signal to Noise Ratio in dB (SNRdB)
        signal_power : power of signal
        noise_power : power of noise
    """
    s = signal_to_noise_ratio(signal_power, noise_power)
    return 10*numpy.log10(s)




def snrdb_to_snr(snr_db: float) -> float:
    """Convert SNRdB to SNR

    snr_{dB} = 20 * log_{10}(snr)
    
    snr_{dB} / 20 = log_{10}(snr)
    
    snr = 10^{snr_{dB} / 20}
    """
    return 10**(snr_db/20)


def snr_to_snrdb(snr: float) -> float:
    """Convert SNR to SNRdB

    snr = 10^{snr_{dB} / 20}
    
    log_{10}(snr) = snr_{dB} / 20
    
    snr_{dB} = 20 *  log_{10}(snr)

    """
    return 20*numpy.log10(snr)

snr = signal_to_noise_ratio
snrdb = signal_to_noise_ratio_db

# def snr(signal_power: float, noise_power: float) -> float:
#     """Compute 
#         signal_power : power of signal
#         noise_power : power of noise
#     """
#     return signal_power / noise_power


# def snrdb(signal_power: float, noise_power: float) -> float:
#     """
#         signal_power : power of signal
#         noise_power : power of noise
#     """
#     s = snr(signal_power, noise_power)
#     return snr_to_snrdb(s)


# def rgb_to_grey_level(image: numpy.ndarray, method: GreyMethod) -> numpy.ndarray:
#     red = image[:, :, 0]
#     green = image[:, :, 1]
#     blue = image[:, :, 2]
#     return 0.299*red + 0.537*green + 0.114*blue