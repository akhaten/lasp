import numpy
import numpy.linalg


class Window2D:

    def __init__(self, array: numpy.ndarray, shape: tuple[int, int]) -> None:
        
        self.array = array
        self.height, self.width = shape

        self.i_min = 0
        self.j_min = 0
        self.i_max = self.array.shape[0] - self.height + 1
        self.j_max = self.array.shape[1] - self.width + 1

        # self.position = self.height // 2, self.width // 2
        # self.i_min = self.height // 2
        # self.i_max = self.array.shape[0] - self.i_min
        # self.j_min = self.width // 2
        # self.j_max = self.array.shape[0] - self.j_min
        
    def __getitem__(self, indices) -> numpy.ndarray:
        i, j = indices
        return self.array[i:i+self.height, j:j+self.width]

    def __iter__(self) -> numpy.ndarray:
        for i in range(self.i_min, self.i_max):
            for j in range(self.j_min, self.j_max):
                yield self.__getitem__((i, j))


# def hard(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
#     expr = numpy.abs(signal)-epsilon
#     return numpy.sign(signal) * numpy.where(0 < expr, expr, 0)


def soft(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    """sign(x) * max(|x| - epsilon, 0)
    """
    expr = numpy.abs(signal)-epsilon
    return numpy.sign(signal) * numpy.where(0 < expr, expr, 0)


def singular_value(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    """ SVT : Singular Value Thresholding
    """
    # Decomposition
    u, s, vh = numpy.linalg.svd(signal)
    # Thresholding
    singular_value_max = numpy.max(s)
    s[s < epsilon*singular_value_max] = 0.
    # Reconstruction
    res = numpy.dot(u * s, vh)
    return res


def singular_value_soft(signal: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    """ SVT : Singular Value Thresholding
    """
    # Decomposition
    u, s, vh = numpy.linalg.svd(signal)
    # Thresholding with 
    singular_value_max = numpy.max(s)
    s_diag = numpy.diag(s)
    s_diag = soft(s_diag, epsilon)
    s = numpy.diag(s_diag)
    # Reconstruction
    res = numpy.dot(u * s, vh)
    return res