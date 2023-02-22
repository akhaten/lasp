import numpy

def signal_to_noise_ratio(signal_power: float, noise_power: float) -> float:
    """
        signal_power : power of signal
        noise_power : power of noise
    """
    return signal_power / noise_power

def signal_to_noise_ratio_db(signal_power: float, noise_power: float) -> float:
    """
        signal_power : power of signal
        noise_power : power of noise
    """
    snr = signal_to_noise_ratio(signal_power, noise_power)
    return 10*numpy.log10(snr)

def snrdb_to_snr(snr_db: float) -> float:
    return 10**(snr_db/10)