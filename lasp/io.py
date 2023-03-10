import matplotlib.pyplot
import matplotlib.cm
import numpy
import pathlib

def read(image_path: pathlib.Path) -> numpy.ndarray:

    ext = image_path.suffix
    image: numpy.ndarray = None

    if ext == '.npy':
        image = numpy.load(image_path)
    else:
        image = matplotlib.pyplot.imread(image_path)

    return image


# def plot_images(grid: numpy.ndarray, titles: numpy.ndarray, cmap: str) -> None:
    
#     m, n = grid.shape
#     idx = 1
#     for i in range(0, m):
#         for j in range(0, n):
#             title = titles[i, j]
#             image = grid[i, j]
#             if not (image is None):
#                 matplotlib.pyplot.subplot(m, n, idx)
#                 matplotlib.pyplot.title(title)
#                 matplotlib.pyplot.imshow(image, cmap)
#             idx += 1

