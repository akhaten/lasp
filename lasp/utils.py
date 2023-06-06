import numpy
import enum

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


def fourier_diagonalization(kernel: numpy.ndarray, shape_out: numpy.ndarray) -> numpy.ndarray:
    """Diagonalize input in Fourier space

    Params:
        - kernel: filter/kernel for diagonalization
        - shape_out: dimesion of output

    Returns:
        Diagonalisation in Fourier space (Complex Array) of kernel with dimension shape out
    """
    nb_rows, nb_cols = kernel.shape
    kernel_padded = numpy.zeros(shape_out)
    kernel_padded[:nb_rows, :nb_cols] = numpy.copy(kernel)
    center = numpy.round(numpy.array(kernel.shape)/2).astype(int) 
    circshifted = circshift(kernel_padded, 1-center)
    return numpy.fft.fft2(circshifted)


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



    