import numpy

import lasp.utils.metrics

def euclidean(signal: numpy.ndarray) -> float:
    return numpy.sqrt(lasp.utils.metrics.power(signal))   