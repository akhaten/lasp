import numpy
import matplotlib.pyplot


def histogram(image_grayscale: numpy.ndarray) -> numpy.ndarray:
    #TODO : TEST
    return numpy.histogram(image_grayscale, bins=256, range=(0, 255))[0]

# def histogram_egalization(histogram: numpy.ndarray, nb_pixel: int) -> numpy.ndarray:
#     pass

def dynamic_rescale(image_grayscale: numpy.ndarray, inf_sup: tuple[int, int]) -> numpy.ndarray:
    
    #TODO: TEST
    a, b = inf_sup
    
    mask_between = (a <= image_grayscale ) and (image_grayscale <= b)
    mask_lt_a = image_grayscale < a
    mask_gt_b = b < image_grayscale

    res = numpy.zeros_like(image_grayscale)
    res[mask_lt_a] = 0
    res[mask_gt_b] = 255
    res[mask_between] = 255 * ((res[mask_between] - a) / (b - a))

    return res

def increase_contrast(image_grayscale: numpy.ndarray, inf_sup: tuple[int, int]) -> numpy.ndarray:

    #TODO: TEST
    a, b = inf_sup
    
    mask_between_0_a = (0 <= image_grayscale ) and (image_grayscale <= a)
    mask_a_255 = image_grayscale < a

    res = numpy.zeros_like(image_grayscale)
    res[mask_between_0_a] = (b / a) * res[mask_between_0_a]
    res[mask_a_255] = ((255-b)*res[mask_a_255] + 255*(b-a)) / (255-a)

    return res
    
    


def sparse(shape: tuple[int, int], epsilon: float) -> numpy.ndarray:
    m, n = shape
    nb_pixel_to_one = int((1-epsilon)*m*n)
    d: numpy.ndarray = numpy.zeros(shape=m*n)
    d[0:nb_pixel_to_one] = 1
    d = numpy.random.permutation(d)
    return numpy.reshape(d, (m, n))


