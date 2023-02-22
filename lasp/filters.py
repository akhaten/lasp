import numpy
import numpy.linalg



# def bilateral_filter(image: numpy.ndarray, sigma_d: float, sigma_r: float, size: int) -> numpy.ndarray:

#     def weight(i: int, j: int, k: int, l: int) -> float:
#         expr1: float = ( (i-k)**2 + (j-l)**2 ) / ( 2*sigma_d**2 )
#         expr2: float = ((image[i, j]-image[k, l]) ** 2)/ (2*sigma_r**2)
#         return numpy.exp(-expr1-expr2)

#     N, M = image.shape
#     half_size = size // 2

#     image_d = numpy.zeros_like(image)
    
#     for i in range(half_size, N-half_size+1):
#         for j in range(half_size, M-half_size+1):
#             sum_iw = 0.0
#             sum_w = 0.0
#             for k in range(i-half_size, i-half_size+1):
#                 for l in range(j-half_size, j-half_size+1):
#                     w = weight(i, j, k, l)
#                     sum_iw += image[k, l] * w
#                     sum_w += w
            
#             if(sum_w == 0.0):
#                 print(sum_iw, sum_w, sum_iw / sum_w)

#             # # print(sum_iw, sum_w)
#             # if(not isinstance(sum_iw, numpy.float64)):
#             #     print(type(sum_iw))
#             #     raise AssertionError("not numpy.float64")
#             # image_d[i, j] = sum_iw / sum_w

#     return image_d


# def bilateral_filter(image: numpy.ndarray, sigma_d: float, sigma_r: float, n_w: int, n_h: int) -> numpy.ndarray:

#     def weight(i: int, j: int, k: int, l: int) -> float:
#         expr1: float = ( (i-k)**2 + (j-l)**2 ) / ( 2*sigma_d**2 )
#         expr2: float = numpy.linalg.norm(image[i, j]-image[k, l], ord=2) ** 2 / (2*sigma_r**2)
#         return numpy.exp(-expr1-expr2)

#     N, M = image.shape
#     image_d = numpy.zeros_like(image)
#     for i in range(0+n_w, N-n_w):
#         for j in range(0+n_h, M-n_h):
#             sum_iw = 0.
#             sum_w = 0.
#             for k in range(-n_w, n_w):
#                 for l in range(-n_h, n_h):
#                     w = weight(i, j, k, l)
#                     sum_iw += image[k, l]*w
#                     sum_w += w
#             image_d[i, j] = sum_iw / sum_w

#     return image_d


def gaussian_filter(size: int, sigma: float, normalize: bool = False) -> numpy.ndarray:

    def gaussian2d_psf(sigma: float, x: float | numpy.ndarray, y: float | numpy.ndarray) -> float | numpy.ndarray:
        exp = numpy.exp( - (x**2+y**2) / (2*sigma**2) ) 
        return ( 1 / (2*numpy.pi*sigma**2) ) * exp


    # half_size = size // 2
    # filter = numpy.zeros((size, size))
    # for x in range(-half_size, half_size+1):
    #     for y in range(-half_size, half_size+1):
    #         filter[half_size+x, half_size+y] = gaussian2d_psf(sigma, x, y)

    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    filter = gaussian2d_psf(sigma, x, y)
    
    if normalize:
        filter /= numpy.sum(filter)

    return filter


def wiener() -> numpy.ndarray:
    pass

def inverse() -> numpy.ndarray:
    pass

def pseudo_inverse() -> numpy.ndarray:
    pass
