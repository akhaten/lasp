import numpy

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

          
def circshift(matrix: numpy.ndarray, shift: numpy.ndarray) -> numpy.ndarray:
    return numpy.roll(matrix, shift, [0, 1])