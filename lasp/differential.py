import numpy
import typing

def dx(image: numpy.ndarray) -> numpy.ndarray:

    nb_rows, nb_cols = numpy.shape(image)
    image_derivated = numpy.zeros(shape=(nb_rows, nb_cols))

    image_derivated[:, 1:nb_cols] = \
        image[:, 1:nb_cols] - image[:, 0:nb_cols-1]

    image_derivated[:, 0] = image[:, 0] - image[:, nb_cols-1]

    return image_derivated
    

def dy(image: numpy.ndarray) -> numpy.ndarray:
    
    nb_rows, nb_cols = numpy.shape(image)
    image_derivated = numpy.zeros(shape=(nb_rows, nb_cols))
    
    image_derivated[1:nb_rows, :] = \
        image[1:nb_rows, :] - image[0:nb_rows-1, :]

    image_derivated[0, :] = image[0, :] - image[nb_rows-1, :]

    return image_derivated


def dxT(image: numpy.ndarray) -> numpy.ndarray:

    nb_rows, nb_cols = numpy.shape(image)
    image_derivated = numpy.zeros(shape=(nb_rows, nb_cols))
    
    image_derivated[:, 0:nb_cols-1] = \
        image[:, 0:nb_cols-1] - image[:, 1:nb_cols]

    image_derivated[:, nb_cols-1] = image[:, nb_cols-1] - image[:, 0]

    return image_derivated


def dyT(image: numpy.ndarray) -> numpy.ndarray:
    
    nb_rows, nb_cols = numpy.shape(image)
    image_derivated = numpy.zeros(shape=(nb_rows, nb_cols))
    
    image_derivated[0:nb_rows-1, :] = \
        image[0:nb_rows-1, :] - image[1:nb_rows, :]

    image_derivated[nb_rows-1, :] = image[nb_rows-1, :] - image[0, :]

    return image_derivated


def differential_matrix(
    derivate: typing.Callable[[numpy.ndarray], numpy.ndarray],
    shape_out: tuple[int, int]
) -> numpy.ndarray:
    # N, M = shape_out
    # return derivate(numpy.eye(N, M))
    nb_rows, nb_cols = shape_out
    return derivate(numpy.eye(nb_rows, nb_cols))
    