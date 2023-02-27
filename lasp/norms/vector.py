import numpy

import lasp.metrics

def euclidean(signal: numpy.ndarray) -> float:
    return numpy.sqrt(lasp.utils.metrics.power(signal))   