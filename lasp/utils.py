import numpy
import enum
import typing


# import pandas

# class Window2D:

#     def __init__(self, array: numpy.ndarray, shape: tuple[int, int]) -> None:
        
#         self.array = array
#         self.height, self.width = shape

#         self.i_min = 0
#         self.j_min = 0
#         self.i_max = self.array.shape[0] - self.height + 1
#         self.j_max = self.array.shape[1] - self.width + 1

#         # self.position = self.height // 2, self.width // 2
#         # self.i_min = self.height // 2
#         # self.i_max = self.array.shape[0] - self.i_min
#         # self.j_min = self.width // 2
#         # self.j_max = self.array.shape[0] - self.j_min
        
#     def __getitem__(self, indices) -> numpy.ndarray:
#         i, j = indices
#         return self.array[i:i+self.height, j:j+self.width]

#     def __iter__(self) -> numpy.ndarray:
#         for i in range(self.i_min, self.i_max):
#             for j in range(self.j_min, self.j_max):
#                 yield self.__getitem__((i, j))


# class DatasetPandas:

#     def __init__(self, columns: list[str]) -> None:
#         self.df = pandas.DataFrame(columns=columns)

#     def 


def pad(array: numpy.ndarray, shape_out: numpy.ndarray | tuple) -> numpy.ndarray:
    
    out_rows, out_cols = shape_out
    nb_rows, nb_cols = array.shape

    pad_rows = out_rows-nb_rows
    pad_cols = out_cols-nb_cols
    return numpy.pad(array, pad_width=[(0, pad_rows), (0, pad_cols)])

          
def circshift(matrix: numpy.ndarray, shift: numpy.ndarray) -> numpy.ndarray:
    """Circular Shift
    Similary to matlab function.

    Params:
        - matrix : matrix
        - shift : shift 

    Returns:
        - Circulary shifted matrix
    """
    return numpy.roll(matrix, shift, [0, 1])


def compute_center(array: numpy.ndarray) -> numpy.ndarray:
    center = numpy.array(array.shape) // 2
    if (center != 0).all():
        center += 1
    return center

def pad_circshift_center(array: numpy.ndarray, shape_out: numpy.ndarray | tuple) -> numpy.ndarray:
    padded = pad(array, shape_out)
    center = compute_center(array)
    circshifted = circshift(padded, 1-center)
    return circshifted


def fourier_diagonalization(kernel: numpy.ndarray, shape_out: numpy.ndarray) -> numpy.ndarray:
    """Diagonalize input in Fourier space

    Params:
        - kernel: filter/kernel for diagonalization
        - shape_out: dimesion of output

    Returns:
        Diagonalisation in Fourier space (Complex Array) of kernel with dimension shape out
    """
    return numpy.fft.fft2(pad_circshift_center(kernel, shape_out))
    # nb_rows, nb_cols = kernel.shape
    # kernel_padded = numpy.zeros(shape_out)
    # kernel_padded[:nb_rows, :nb_cols] = numpy.copy(kernel)


    # center_row = nb_rows // 2
    # center_col = nb_cols // 2
    


    # center = numpy.divide(kernel.shape, 2).astype(numpy.int8) + 1
    # circshifted = circshift(kernel_padded, 1-center)

    # print(numpy.array(kernel.shape) / 2)
    # print(numpy.round(numpy.array(kernel.shape) / 2))
    # center = numpy.array(kernel.shape) // 2
    
    # if (center != 0).all():
    #     # print('diff')
    #     center += 1
    # # center[center != 0] += 1
    # # center = center.astype(numpy.int8)

    # print('center :', center)
    # circshifted = circshift(kernel_padded, 1-center)
    # # circshifted = numpy.roll(kernel_padded, 1-center[0], 1)
    # # circshifted = numpy.roll(kernel_padded, 1-center[1], 0)
    # return circshifted
    # print('circshifted :\n', circshifted)
    # return numpy.fft.fft2(circshifted)


def normalize(img: numpy.ndarray) -> numpy.ndarray:
    """Normalize image

    Normalization formula:

    (img - min(img)) / (max(img) - min(img))

    Params:
        -img: image for normalization

    Return:
        img normalized
    """
    val_min = numpy.min(img)
    val_max = numpy.max(img)
    return (img - val_min) / (val_max - val_min)


def blockproc(
    array: numpy.ndarray, 
    block_shape: numpy.ndarray,
    fun: typing.Callable[[numpy.ndarray], numpy.ndarray]
) -> numpy.ndarray:
    
    nb_rows, nb_cols = array.shape
    step_r, step_c = block_shape
    
    for i in range(0, nb_rows, step_r):
        for j in range(0, nb_cols, step_c):
            array[i:i+step_r, j:j+step_c] = \
                fun(array[i:i+step_r, j:j+step_c])
    
    return array

def blockproc_reshape(
    array: numpy.ndarray,
    block_size: numpy.ndarray,
    order: str
) -> numpy.ndarray:

    nb_rows, nb_cols = array.shape
    res = None

    step_r, step_c = block_size

    for j in range(0, nb_cols, step_c):

        column = None
        
        for i in range(0, nb_rows, step_r):
            
            bloc = numpy.reshape(
                array[i:i+step_r, j:j+step_c], 
                (step_r*step_c, 1),
                order=order
            )
            if column is None:
                column = numpy.copy(bloc)
            else:
                column = numpy.vstack([ column, bloc ])

        if res is None:
            res = numpy.copy(column)
        else:
            res = numpy.hstack([ res, column ])

    return res