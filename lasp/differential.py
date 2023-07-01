import numpy
import typing

def dx(image: numpy.ndarray) -> numpy.ndarray:
    """ Derivation by column

    Params:
        - image
    
    Return:
        - first element of gradient
    """

    nb_rows, nb_cols = numpy.shape(image)
    image_derivated = numpy.zeros(shape=(nb_rows, nb_cols))

    image_derivated[:, 1:nb_cols] = \
        image[:, 1:nb_cols] - image[:, 0:nb_cols-1]

    image_derivated[:, 0] = image[:, 0] - image[:, nb_cols-1]

    return image_derivated
    

def dy(image: numpy.ndarray) -> numpy.ndarray:
    """ Derivation by line

    Params:
        - image
    
    Return:
        - second element of gradient
    """
    
    nb_rows, nb_cols = numpy.shape(image)
    image_derivated = numpy.zeros(shape=(nb_rows, nb_cols))
    
    image_derivated[1:nb_rows, :] = \
        image[1:nb_rows, :] - image[0:nb_rows-1, :]

    image_derivated[0, :] = image[0, :] - image[nb_rows-1, :]

    return image_derivated


def dxT(image: numpy.ndarray) -> numpy.ndarray:
    """ Derivation Transposed by column

    Params:
        - image
    
    Return:
        - first element of gradient transposed
    """

    nb_rows, nb_cols = numpy.shape(image)
    image_derivated = numpy.zeros(shape=(nb_rows, nb_cols))
    
    image_derivated[:, 0:nb_cols-1] = \
        image[:, 0:nb_cols-1] - image[:, 1:nb_cols]

    image_derivated[:, nb_cols-1] = image[:, nb_cols-1] - image[:, 0]

    return image_derivated


def dyT(image: numpy.ndarray) -> numpy.ndarray:
    """ Derivation Transposed by line

    Params:
        - image
    
    Return:
        - second element of gradient transposed
    """
    nb_rows, nb_cols = numpy.shape(image)
    image_derivated = numpy.zeros(shape=(nb_rows, nb_cols))
    
    image_derivated[0:nb_rows-1, :] = \
        image[0:nb_rows-1, :] - image[1:nb_rows, :]

    image_derivated[nb_rows-1, :] = image[nb_rows-1, :] - image[0, :]

    return image_derivated


def derivation(arr: numpy.ndarray, axis: int) -> numpy.ndarray:
    shape = numpy.copy(arr.shape)
    n = shape[axis]
    taken = numpy.take(arr, axis=axis, indices=n-1)
    shape[axis] = 1
    to_add = numpy.reshape(taken, shape)
    return numpy.diff(arr, prepend=to_add, axis=axis)


def kernel_identity(dim: int) -> numpy.array:
    shape = numpy.full(shape=dim, fill_value=1)
    one_nd = numpy.full(shape=tuple(shape), fill_value=1)
    return numpy.pad(array=one_nd, pad_width=1)


def differential_matrix(
    derivate: typing.Callable[[numpy.ndarray], numpy.ndarray],
    shape_out: tuple[int, int]
) -> numpy.ndarray:
    """Create matrix to compute derivation by dot matrix


    Params:
        - derivate : dx, dy, dxT or dyT
        - shape_out : shape of output

    Return:
        - Discrete Difinition with Finite Elements like matrix
    """
    # N, M = shape_out
    # return derivate(numpy.eye(N, M))
    nb_rows, nb_cols = shape_out
    return derivate(numpy.eye(nb_rows, nb_cols))
    